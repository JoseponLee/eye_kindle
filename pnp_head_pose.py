#!/usr/bin/env python

##
# Massimiliano Patacchiola, Plymouth University 2016
#
# This is an example of head pose estimation with solvePnP and dlib face detector.
# It uses the dlib library and openCV.
# To use this example you have to provide an input video file
# and an output path:
# python ex_pnp_pose_estimation_video.py /home/video.mpg ./output.avi
#

import numpy as np
import numpy
import math
import cv2
import sys
import os

from face_landmark_detection import faceLandmarkDetection

# For the frontal face detector
import dlib

# Antropometric constant values of the human head.
# Found on wikipedia and on:
# "Head-and-Face Anthropometric Survey of U.S. Respirator Users"
#
# X-Y-Z with X pointing forward and Y on the left.
# The X-Y-Z coordinates used are like the standard
# coordinates of ROS (robotic operative system)
P3D_RIGHT_SIDE = numpy.float32([-100.0, -77.5, -5.0])  # 0
P3D_GONION_RIGHT = numpy.float32([-110.0, -77.5, -85.0])  # 4
P3D_MENTON = numpy.float32([0.0, 0.0, -122.7])  # 8
P3D_GONION_LEFT = numpy.float32([-110.0, 77.5, -85.0])  # 12
P3D_LEFT_SIDE = numpy.float32([-100.0, 77.5, -5.0])  # 16
P3D_FRONTAL_BREADTH_RIGHT = numpy.float32([-20.0, -56.1, 10.0])  # 17
P3D_FRONTAL_BREADTH_LEFT = numpy.float32([-20.0, 56.1, 10.0])  # 26
P3D_SELLION = numpy.float32([0.0, 0.0, 0.0])  # 27
P3D_NOSE = numpy.float32([21.1, 0.0, -48.0])  # 30
P3D_SUB_NOSE = numpy.float32([5.0, 0.0, -52.0])  # 33
P3D_RIGHT_EYE = numpy.float32([-20.0, -65.5, -5.0])  # 36
P3D_RIGHT_TEAR = numpy.float32([-10.0, -40.5, -5.0])  # 39
P3D_LEFT_TEAR = numpy.float32([-10.0, 40.5, -5.0])  # 42
P3D_LEFT_EYE = numpy.float32([-20.0, 65.5, -5.0])  # 45
# P3D_LIP_RIGHT = numpy.float32([-20.0, 65.5,-5.0]) #48
# P3D_LIP_LEFT = numpy.float32([-20.0, 65.5,-5.0]) #54
P3D_STOMION = numpy.float32([10.0, 0.0, -75.0])  # 62

# The points to track
# These points are the ones used by PnP
# to estimate the 3D pose of the face
TRACKED_POINTS = (0, 4, 8, 12, 16, 17, 26, 27, 30, 33, 36, 39, 42, 45, 62)
ALL_POINTS = list(range(0, 68))  # Used for debug only

video_capture = cv2.VideoCapture(0)


# Obtaining the CAM dimension
cam_w = int(video_capture.get(3))
cam_h = int(video_capture.get(4))

# Defining the camera matrix.
# To have better result it is necessary to find the focal
# lenght of the camera. fx/fy are the focal lengths (in pixels)
# and cx/cy are the optical centres. These values can be obtained
# roughly by approximation, for example in a 640x480 camera:
# cx = 640/2 = 320
# cy = 480/2 = 240
# fx = fy = cx/tan(60/2 * pi / 180) = 554.26
c_x = cam_w / 2
c_y = cam_h / 2
f_x = c_x / numpy.tan(60 / 2 * numpy.pi / 180)
f_y = f_x

# Estimated camera matrix values.
camera_matrix = numpy.float32([[f_x, 0.0, c_x],
                               [0.0, f_y, c_y],
                               [0.0, 0.0, 1.0]])

print("Estimated camera matrix: \n" + str(camera_matrix) + "\n")

# These are the camera matrix values estimated on my webcam with
# the calibration code (see: src/calibration):
# camera_matrix = numpy.float32([[602.10618226,          0.0, 320.27333589],
# [         0.0, 603.55869786,  229.7537026],
# [         0.0,          0.0,          1.0] ])

# Distortion coefficients
camera_distortion = numpy.float32([0.0, 0.0, 0.0, 0.0, 0.0])

# Distortion coefficients estimated by calibration
# camera_distortion = numpy.float32([ 0.06232237, -0.41559805,  0.00125389, -0.00402566,  0.04879263])

# This matrix contains the 3D points of the
# 11 landmarks we want to find. It has been
# obtained from antrophometric measurement
# on the human head.
landmarks_3D = numpy.float32([P3D_RIGHT_SIDE,
                              P3D_GONION_RIGHT,
                              P3D_MENTON,
                              P3D_GONION_LEFT,
                              P3D_LEFT_SIDE,
                              P3D_FRONTAL_BREADTH_RIGHT,
                              P3D_FRONTAL_BREADTH_LEFT,
                              P3D_SELLION,
                              P3D_NOSE,
                              P3D_SUB_NOSE,
                              P3D_RIGHT_EYE,
                              P3D_RIGHT_TEAR,
                              P3D_LEFT_TEAR,
                              P3D_LEFT_EYE,
                              P3D_STOMION])

# Declaring the two classifiers
# my_cascade = haarCascade("../etc/haarcascade_frontalface_alt.xml", "../etc/haarcascade_profileface.xml")
dlib_landmarks_file = "./gaze_tracking/shape_predictor_68_face_landmarks.dat"
my_detector = faceLandmarkDetection(dlib_landmarks_file)
my_face_detector = dlib.get_frontal_face_detector()


def rotationMatrixToEulerAngles(R):
    # assert(isRotationMatrix(R))

    # To prevent the Gimbal Lock it is possible to use
    # a threshold of 1e-6 for discrimination
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])

def get_head_pose(frame, face):
    face_x1 = face.left()
    face_y1 = face.top()
    face_x2 = face.right()
    face_y2 = face.bottom()
    face_center = np.array([(face_x1 + face_x2) / 2, (face_y1 + face_y2) / 2])

    landmarks_2D = my_detector.returnLandmarks(frame, face_x1, face_y1, face_x2, face_y2,
                                               points_to_return=TRACKED_POINTS)
    retval, rvec, tvec = cv2.solvePnP(landmarks_3D,
                                      landmarks_2D,
                                      camera_matrix, camera_distortion)

    # Now we project the 3D points into the image plane
    # Creating a 3-axis to be used as reference in the image.
    axis = numpy.float32([[50, 0, 0],
                          [0, 50, 0],
                          [0, 0, 50]])
    imgpts, jac = cv2.projectPoints(axis, rvec, tvec, camera_matrix, camera_distortion)

    # Drawing the three axis on the image frame.
    # The opencv colors are defined as BGR colors such as:
    # (a, b, c) >> Blue = a, Green = b and Red = c
    # Our axis/color convention is X=R, Y=G, Z=B
    sellion_xy = (landmarks_2D[7][0], landmarks_2D[7][1])

    rmat, _ = cv2.Rodrigues(rvec)
    pos = rotationMatrixToEulerAngles(rmat)
    # print(pos)
    return pos, imgpts, sellion_xy, face_center


# def main():
#     # Create the main window and move it
#     cv2.namedWindow('Video')
#     cv2.moveWindow('Video', 20, 20)
#
#     while (True):
#
#         # Capture frame-by-frame
#         ret, frame = video_capture.read()
#         # gray = cv2.cvtColor(frame[roi_y1:roi_y2, roi_x1:roi_x2], cv2.COLOR_BGR2GRAY)
#
#         faces_array = my_face_detector(frame, 0)
#         print("Total Faces: " + str(len(faces_array)))
#         for i, pos in enumerate(faces_array):
#
#             face_x1 = pos.left()
#             face_y1 = pos.top()
#             face_x2 = pos.right()
#             face_y2 = pos.bottom()
#             text_x1 = face_x1
#             text_y1 = face_y1 - 3
#
#             cv2.putText(frame, "FACE " + str(i + 1), (text_x1, text_y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1);
#             cv2.rectangle(frame,
#                           (face_x1, face_y1),
#                           (face_x2, face_y2),
#                           (0, 255, 0),
#                           2)
#
#             landmarks_2D = my_detector.returnLandmarks(frame, face_x1, face_y1, face_x2, face_y2,
#                                                        points_to_return=TRACKED_POINTS)
#
#             for point in landmarks_2D:
#                 cv2.circle(frame, (point[0], point[1]), 2, (0, 0, 255), -1)
#
#             # Applying the PnP solver to find the 3D pose
#             # of the head from the 2D position of the
#             # landmarks.
#             # retval - bool
#             # rvec - Output rotation vector that, together with tvec, brings
#             # points from the model coordinate system to the camera coordinate system.
#             # tvec - Output translation vector.
#             retval, rvec, tvec = cv2.solvePnP(landmarks_3D,
#                                               landmarks_2D,
#                                               camera_matrix, camera_distortion)
#
#             # Now we project the 3D points into the image plane
#             # Creating a 3-axis to be used as reference in the image.
#             axis = numpy.float32([[50, 0, 0],
#                                   [0, 50, 0],
#                                   [0, 0, 50]])
#             imgpts, jac = cv2.projectPoints(axis, rvec, tvec, camera_matrix, camera_distortion)
#
#             # Drawing the three axis on the image frame.
#             # The opencv colors are defined as BGR colors such as:
#             # (a, b, c) >> Blue = a, Green = b and Red = c
#             # Our axis/color convention is X=R, Y=G, Z=B
#             sellion_xy = (landmarks_2D[7][0], landmarks_2D[7][1])
#             cv2.line(frame, sellion_xy, tuple(imgpts[1].ravel()), (0, 255, 0), 3)  # GREEN
#             cv2.line(frame, sellion_xy, tuple(imgpts[2].ravel()), (255, 0, 0), 3)  # BLUE
#             cv2.line(frame, sellion_xy, tuple(imgpts[0].ravel()), (0, 0, 255), 3)  # RED
#
#             rmat, _ = cv2.Rodrigues(rvec)
#             pos = rotationMatrixToEulerAngles(rmat)
#             print(pos)
#
#             # Writing in the output file
#
#         # Showing the frame and waiting
#         # for the exit command
#         cv2.imshow('Video', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'): break
#
#     # Release the camera
#     video_capture.release()
#     print("Bye...")


# if __name__ == "__main__":
#     main()
