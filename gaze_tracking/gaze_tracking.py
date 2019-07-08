from __future__ import division
from keras import models
import os
import time
import cv2
import dlib
import pnp_head_pose
from .eye import Eye
from .calibration import Calibration
import numpy as np
import pyautogui as pag
import pyperclip as pcp
from pykalman import KalmanFilter
import train
import speech

t = 0
kf = KalmanFilter(transition_matrices=np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]]),
                  observation_matrices=np.array([[1, 0, 0, 0], [0, 1, 0, 0]]),
                  transition_covariance=0.03 * np.eye(4))
def kalman(x, y):
    global t, filtered_state_means0, filtered_state_covariances0, lmx, lmy, lpx, lpy
    current_measurement = np.array([np.float32(x), np.float32(y)])
    # cmx, cmy = current_measurement[0], current_measurement[1]
    if t == 0:
        filtered_state_means0 = np.array([0.0, 0.0, 0.0, 0.0])
        filtered_state_covariances0 = np.eye(4)
        lmx, lmy = 0.0, 0.0
        lpx, lpy = 0.0, 0.0
    filtered_state_means, filtered_state_covariances = (
        kf.filter_update(filtered_state_means0, filtered_state_covariances0, current_measurement))
    cpx, cpy = filtered_state_means[0], filtered_state_means[1]
    filtered_state_means0, filtered_state_covariances0 = filtered_state_means, filtered_state_covariances
    t = t + 1
    lpx, lpy = filtered_state_means[0], filtered_state_means[1]
    lmx, lmy = current_measurement[0], current_measurement[1]
    return cpx, cpy


flag = [0, 0, 0, 0, 0]
left = 200
right = 1800
up = 150
down = 850
cnt = 7
def stay_judge(x, y):
    global flag
    if x < left and y > up and y < down:
        if flag[4]:
            flag = [1, 0, 0, 0, 0]
        else:
            flag[0] += 1
            if flag[0] >= cnt:
                flag[0] = 0
                pag.press('left')

    elif x > right and y > up and y < down:
        if flag[4]:
            flag = [0, 1, 0, 0, 0]
        else:
            flag[1] += 1
            if flag[1] >= cnt:
                flag[1] = 0
                pag.press('right')

    elif y < up and x > left and x < right:
        if flag[4]:
            flag = [0, 0, 1, 0, 0]
        else:
            flag[2] += 1
            if flag[2] >= cnt:
                flag[2] = 0
                for time in range(10):
                    pag.press('up')


    elif y > down and x > left and x < right:
        if flag[4]:
            flag = [0, 0, 0, 1, 0]
        else:
            flag[3] += 1
            if flag[3] >= cnt:
                flag[3] = 0
                for time in range(10):
                    pag.press('down')


    else:
        flag = [0, 0, 0, 0, 1]

def distance(pos1, pos2):
    return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

class GazeTracking(object):
    """
    This class tracks the user's gaze.
    It provides useful information like the position of the eyes
    and pupils and allows to know if the eyes are open or closed
    """

    def __init__(self):
        self.frame = None
        self.eye_left = None
        self.eye_right = None
        self.calibration = Calibration()
        self.roll_pitch_yaw = None
        self.imgpts = None
        self.sellion_xy = None
        self.pupil = None
        self.face = None
        self.data = []
        self.label = []
        self.result = []
        self.mouse_pos = None
        self.landmarks = None
        self.mouse_flag = False

        # _face_detector is used to detect faces
        self._face_detector = dlib.get_frontal_face_detector()

        # _predictor is used to get facial landmarks of a given face
        cwd = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.abspath(os.path.join(cwd, "shape_predictor_68_face_landmarks.dat"))
        self._predictor = dlib.shape_predictor(model_path)
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.loaded_model = models.model_from_json(loaded_model_json)
        # load weights into new model
        self.loaded_model.load_weights("model.h5")
        print("Loaded model from disk")

    @property
    def pupils_located(self):
        """Check that the pupils have been located"""
        try:
            int(self.eye_left.pupil.x)
            int(self.eye_left.pupil.y)
            int(self.eye_right.pupil.x)
            int(self.eye_right.pupil.y)
            return True
        except Exception:
            return False

    def _analyze(self):
        """Detects the face and initialize Eye objects"""
        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_detector(frame)

        try:
            self.landmarks = self._predictor(frame, faces[0])
            self.face_width = faces[0].right() - faces[0].left()
            self.roll_pitch_yaw, self.imgpts, self.sellion_xy, self.face = pnp_head_pose.get_head_pose(frame, faces[0])
            self.mouse_pos = pag.position()  # 返回鼠标的坐标
            # print(self.roll_pitch_yaw, self.face)
            self.eye_left = Eye(frame, self.landmarks, 0, self.calibration)
            self.eye_right = Eye(frame, self.landmarks, 1, self.calibration)

        except IndexError:
            self.eye_left = None
            self.eye_right = None
            self.roll_pitch_yaw, self.imgpts, self.sellion_xy, self.face, self.mouse_pos = None, None, None, None, None

    def HighLight_judge(self, thu):
        if len(self.result) <= 20:
            pass
        else:
            # print(distance(self.result[-1], self.result[-5]), distance(self.result[-1], self.result[-10]))
            if distance(self.result[-1], self.result[-5]) < thu and distance(self.result[-1], self.result[-10]) < thu\
                    and distance(self.result[-1], self.result[-15]) < thu and distance(self.result[-1], self.result[-20]) < thu:
                pag.doubleClick()
                pag.hotkey('ctrl', '9')
                self.result = []

    def mouse_judge(self):
        filename = '01.wav'
        rate = 16000
        mouth_higth = (self.landmarks.part(66).y - self.landmarks.part(62).y) / self.face_width  # 嘴巴张开程度
        if mouth_higth > 0.08 and self.mouse_flag == False:
            self.mouse_flag = True
            pag.hotkey('ctrl', '6')
            speech.recording(filename, rate)
            token = speech.get_token()
            signal = open(filename, "rb").read()
            ans = speech.recognize(signal, rate, token)
            if ans:
                pcp.copy(ans)
                pag.hotkey('ctrl', 'v')
                print(ans)
            pag.click()
        elif mouth_higth < 0.06:
            self.mouse_flag = False

    def state(self):
        if self.pupil is not None and self.roll_pitch_yaw is not None and self.mouse_pos is not None:
            X_test = np.array([np.hstack((self.pupil, self.roll_pitch_yaw, self.face))])
            ans = train.test(self.loaded_model, X_test)
            res = ans * 100
            # x = (res[0, 0] - left_x) * kx
            # y = (res[0, 1] - up_y) * ky
            x, y = kalman(res[0, 0], res[0, 1])
            if x >= 1919: x = 1918
            elif x <= 0: x = 1
            if y >= 1079: y = 1078
            elif y <= 0: y = 1
            # print(x, y)
            self.result.append((x, y))
            pag.moveTo(x, y)
            stay_judge(x, y)
            self.HighLight_judge(thu=120)
            self.mouse_judge()

    def make_data(self):
        if self.pupil is not None and self.roll_pitch_yaw is not None and self.mouse_pos is not None:
            self.data.append(np.hstack((self.pupil, self.roll_pitch_yaw, self.face)))
            self.label.append(np.array(self.mouse_pos))

    def save_data(self):
        np.savez('data_train1.npz', train_data=np.array(self.data), train_label=np.array(self.label))
        # np.savez('data_val1.npz', val_data=np.array(self.data), val_label=np.array(self.label))

    def refresh(self, frame):
        """Refreshes the frame and analyzes it.

        Arguments:
            frame (numpy.ndarray): The frame to analyze
        """
        self.frame = frame
        self._analyze()

    def pupil_left_coords(self):
        """Returns the coordinates of the left pupil"""
        if self.pupils_located:
            x = self.eye_left.origin[0] + self.eye_left.pupil.x
            y = self.eye_left.origin[1] + self.eye_left.pupil.y
            return (x, y)

    def pupil_right_coords(self):
        """Returns the coordinates of the right pupil"""
        if self.pupils_located:
            x = self.eye_right.origin[0] + self.eye_left.pupil.x
            y = self.eye_right.origin[1] + self.eye_left.pupil.y
            return (x, y)

    def horizontal_ratio(self):
        """Returns a number between 0.0 and 1.0 that indicates the
        horizontal direction of the gaze. The extreme right is 0.0,
        the center is 0.5 and the extreme left is 1.0
        """
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.x / (self.eye_left.center[0] * 2 - 10)
            pupil_right = self.eye_right.pupil.x / (self.eye_right.center[0] * 2 - 10)
            return (pupil_left + pupil_right) / 2

    def vertical_ratio(self):
        """Returns a number between 0.0 and 1.0 that indicates the
        vertical direction of the gaze. The extreme top is 0.0,
        the center is 0.5 and the extreme bottom is 1.0
        """
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.y / (self.eye_left.center[1] * 2 - 10)
            pupil_right = self.eye_right.pupil.y / (self.eye_right.center[1] * 2 - 10)
            return (pupil_left + pupil_right) / 2

    def is_right(self):
        """Returns true if the user is looking to the right"""
        if self.pupils_located:
            return self.horizontal_ratio() <= 0.35

    def is_left(self):
        """Returns true if the user is looking to the left"""
        if self.pupils_located:
            return self.horizontal_ratio() >= 0.65

    def is_center(self):
        """Returns true if the user is looking to the center"""
        if self.pupils_located:
            return self.is_right() is not True and self.is_left() is not True

    def is_blinking(self):
        """Returns true if the user closes his eyes"""
        if self.pupils_located:
            blinking_ratio = (self.eye_left.blinking + self.eye_right.blinking) / 2
            return blinking_ratio > 3.8

    def annotated_frame(self):
        """Returns the main frame with pupils highlighted"""
        frame = self.frame.copy()

        if self.pupils_located:
            color = (0, 255, 0)
            x_left, y_left = self.pupil_left_coords()
            x_right, y_right = self.pupil_right_coords()
            self.pupil = np.array([x_left, y_left, x_right, y_right])
            # print("pupil", self.pupil)
            cv2.line(frame, (x_left - 5, y_left), (x_left + 5, y_left), color)
            cv2.line(frame, (x_left, y_left - 5), (x_left, y_left + 5), color)
            cv2.line(frame, (x_right - 5, y_right), (x_right + 5, y_right), color)
            cv2.line(frame, (x_right, y_right - 5), (x_right, y_right + 5), color)
            cv2.line(frame, self.sellion_xy, tuple(self.imgpts[1].ravel()), (0, 255, 0), 3)  # GREEN
            cv2.line(frame, self.sellion_xy, tuple(self.imgpts[2].ravel()), (255, 0, 0), 3)  # BLUE
            cv2.line(frame, self.sellion_xy, tuple(self.imgpts[0].ravel()), (0, 0, 255), 3)  # RED
        else:
            self.pupil = None

        return frame
