import cv2
import numpy as np
from pykalman import KalmanFilter

frame = np.zeros((800, 800, 3), np.uint8)
kf = KalmanFilter(transition_matrices=np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]]),
                  observation_matrices=np.array([[1, 0, 0, 0], [0, 1, 0, 0]]),
                  transition_covariance=0.03 * np.eye(4))
# transition_matrices：公式中的A
# observation_matrices：公式中的H
# transition_covariance：公式中的Q

t = 0

# 状态值为x_t=[x,y,dx,dy],其中(x,y)为鼠标当前位置，（dx,dy）指速度分量
# 直接获得的观测为位置(x,y)
def mousemove(event, x, y, s, p):
    global t, filtered_state_means0, filtered_state_covariances0, lmx, lmy, lpx, lpy
    current_measurement = np.array([np.float32(x), np.float32(y)])
    cmx, cmy = current_measurement[0], current_measurement[1]
    if t == 0:
        filtered_state_means0 = np.array([0.0, 0.0, 0.0, 0.0])
        filtered_state_covariances0 = np.eye(4)
        lmx, lmy = 0.0, 0.0
        lpx, lpy = 0.0, 0.0

    filtered_state_means, filtered_state_covariances = (kf.filter_update(filtered_state_means0, filtered_state_covariances0, current_measurement))
    cpx, cpy = filtered_state_means[0], filtered_state_means[1]
    # 绘制测量值轨迹（绿色）
    cv2.line(frame, (int(lmx), int(lmy)), (int(cmx), int(cmy)), (0, 100, 0))
    # 绘制预测值轨迹（红色）
    cv2.line(frame, (int(lpx), int(lpy)), (int(cpx), int(cpy)), (0, 0, 200))
    filtered_state_means0, filtered_state_covariances0 = filtered_state_means, filtered_state_covariances
    t = t + 1
    lpx, lpy = filtered_state_means[0], filtered_state_means[1]
    lmx, lmy = current_measurement[0], current_measurement[1]


cv2.namedWindow("kalman_tracker")
cv2.setMouseCallback("kalman_tracker", mousemove)
while True:
    cv2.imshow("kalman_tracker", frame)
    if (cv2.waitKey(30) & 0xff) == 27:
        break

cv2.destroyAllWindows()
