import cv2 as cv
import numpy as np

LOWER_RED = np.array([int(1 / 360 * 255), 110, int(0.6 * 255)])
UPPER_RED = np.array([int(0.6 * 255), 255, 255])
KERNEL5B5 = np.ones((5, 5), np.uint8)
KERNEL8B8 = np.ones((9, 9), np.uint8)


def track_ball(video_path: str):
    cap = cv.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            img_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            mask = cv.inRange(img_hsv, LOWER_RED, UPPER_RED, )
            mask = cv.dilate(mask, KERNEL5B5, iterations=1)
            mask = cv.morphologyEx(mask, cv.MORPH_OPEN, KERNEL5B5)
            mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, KERNEL8B8)
            ret, tresh = cv.threshold(mask, 127, 255, 0)
            m = cv.moments(tresh)
            cx = int(m['m10'] / m['m00'])
            cy = int(m['m01'] / m['m00'])
            cv.circle(frame, (cx, cy), 10, (255, 0, 0), -1)
            frame = cv.resize(frame, [800, 800])
            cv.imshow('Frame', frame)
            if cv.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv.destroyAllWindows()


track_ball('./movingball.mp4')
