import numpy as np
import cv2 as cv
from playsound import playsound
import time

SOUND_INPUT = './input/sounds/'
OUTPUT = './output/'
PER_CLASS = 100
COUNTER = 0

def startCapture(capture, section):
    global COUNTER
    while True:
        ret, frame = capture.read()
        if not ret or np.sum(frame) == 0:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        cv.imwrite(OUTPUT + str(COUNTER + 1) + '_' + str(section) + '.jpg', frame)
        COUNTER += 1
        if cv.waitKey(1) == ord('q') or COUNTER % PER_CLASS == 0:
            break


def getCapturer():
    capture = False
    while(not capture):
        capture = cv.VideoCapture(1)
        if not capture.isOpened():
            print("Cannot open camera")
            exit()

        ret, frame = capture.read()
        if not ret or np.sum(frame) == 0:
            print("Can't receive frame (stream end?). Exiting ...")
            capture.release()
            capture = False
    return capture

if __name__ == '__main__':
    capture = getCapturer()
    time.sleep(2)
    
    for i in range(9):
        for j in range(3):
            playsound(SOUND_INPUT + 'countdown.mp3')
            time.sleep(0.2)
        playsound(SOUND_INPUT + 'start.mp3')
        startCapture(capture, i + 1)
        playsound(SOUND_INPUT + 'stop.mp3')
        time.sleep(3)

    capture.release()
    cv.destroyAllWindows()