"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import cv2
from gaze_tracking import GazeTracking

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)
flag = 0

while True:
    # We get a new frame from the webcam
    _, frame = webcam.read()
    # cv2.imwrite("img.jpg", frame)

    # We send this frame to GazeTracking to analyze it
    gaze.refresh(frame)

    frame = gaze.annotated_frame()

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (0, 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (0, 40), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 1)

    if flag == 1:
        gaze.make_data()

    cv2.imshow("Demo", frame)

    if cv2.waitKey(1) == 27:
        break

    elif cv2.waitKey(1) == ord("t"):
        # 按t开始
        print("start")
        flag = 1

gaze.save_data()