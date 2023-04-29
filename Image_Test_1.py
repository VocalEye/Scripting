import os
import numpy as np
import cv2 as cv

PHOTO_INPUT_PATH = '../Randomized Camera Dataset #1/1/imgs/'
PHOTO_OUTPUT_PATH = './output/'
CLASSIFIER_PATH = './models/haarcascade_eye.xml'
# CLASSIFIER_PATH = './models/haarcascade_eye_tree_eyeglasses.xml'

MINIMUM_NEIGHBORS = 10

imageCount = len(os.listdir(PHOTO_INPUT_PATH))
print(imageCount)
report = open(PHOTO_OUTPUT_PATH + "report.csv", "w")
eyeCascade = cv.CascadeClassifier(CLASSIFIER_PATH)

def saveImage(image, point1, point2, index):
    cutImage = image[point1[1]:point2[1], point1[0]:point2[0]]
    cv.imwrite(PHOTO_OUTPUT_PATH + str(index) + '.jpg', cutImage)

def mainExecution():
    for i in range(1, imageCount+1):
        image = cv.imread(PHOTO_INPUT_PATH + str(i) + '.jpg')
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        eyes = eyeCascade.detectMultiScale(gray, minNeighbors = MINIMUM_NEIGHBORS)
        
        if len(eyes) == 1:
            (x, y, w, h) = eyes[0]
            saveImage(image, (x, y), (x + w, y + h), i)
        elif len(eyes) < 1:
            report.write(PHOTO_OUTPUT_PATH + str(i) + '.jpg,0,[]\n')
        else:
            print(len(eyes))
            (x, y, w, h) = eyes[0]
            saveImage(image, (x, y), (x + w, y + h), i)
            
            stringifiedEyes = str(eyes)
            report.write(PHOTO_OUTPUT_PATH + str(i) + '.jpg,'+ str(len(eyes)) + ',[' + stringifiedEyes + ']\n' )
    report.close()
    cv.destroyAllWindows()

mainExecution()