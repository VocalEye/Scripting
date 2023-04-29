import cv2 as cv
import numpy as np
import json
from glob import glob
import random
import os

IMAGE_SIZE = (600, 400)
PHOTO_INPUT_PATH = '../Randomized Camera Dataset #2/'
PHOTO_OUTPUT_PATH = './output/'

def processJsonList(image, jsonList):
    landmarks = [eval(s) for s in jsonList]
    return np.array([(x, image.shape[0]-y, z) for (x, y, z) in landmarks])

def drawPointsAndLines(image, innerMargin, caruncle, iris):
    # Draw black background points and lines
    for landmark in np.vstack([innerMargin, caruncle, iris[::2]]):
        cv.circle(image, (int(landmark[0]), int(landmark[1])), 3, (0, 0, 0), -1)
        cv.polylines(image, np.array(
            [innerMargin[:, :2]], int), True, (0, 0, 0), 2)
        cv.polylines(image, np.array(
            [iris[:, :2]], int), True, (0, 0, 0), 2)

    # Draw green foreground points and lines
    for landmark in np.vstack([innerMargin, caruncle, iris[::2]]):
        cv.circle(image, (int(landmark[0]), int(landmark[1])), 2, (0, 255, 0), -1)
        cv.polylines(image, np.array(
            [innerMargin[:, :2]], int), True, (0, 255, 0), 1)
        cv.polylines(image, np.array(
            [iris[:, :2]], int), True, (0, 255, 0), 1)

def drawLookVector(image, lookVector, eyeCenter):
    lookVector[1] = -lookVector[1]
    strangeValue = tuple(eyeCenter + ( np.array(lookVector[:2]) * 80).astype(int))
    cv.line(image, tuple(eyeCenter), strangeValue, (0, 0, 0), 3)
    cv.line(image, tuple(eyeCenter), strangeValue, (0, 255, 255), 2)
    
def cropImageAround(image, eyeCenter, imageSize):
    widthHalf = imageSize[0] / 2
    heightHalf = imageSize[1] / 2
    firstPoint = (int(eyeCenter[0] - widthHalf), int(eyeCenter[1] - heightHalf))
    secondPoint = (int(eyeCenter[0] + widthHalf), int(eyeCenter[1] + heightHalf))
    
    return image[firstPoint[1]:secondPoint[1], firstPoint[0]:secondPoint[0]]

def findCenterOfImage(innerMargin, caruncle):
    innerMarginX = list(map(lambda margin: margin[0], innerMargin))
    innerMarginY = list(map(lambda margin: margin[1], innerMargin))
    caruncleX = list(map(lambda can: can[0], caruncle))

    innerMarginX.sort()
    innerMarginY.sort()
    caruncleX.sort()

    firstPoint = (caruncleX[0], innerMarginY[0])
    secondPoint = (innerMarginX[len(innerMarginX)-1], innerMarginY[len(innerMarginY)-1])

    return [
        firstPoint[0] + (secondPoint[0] - firstPoint[0])/2,
        firstPoint[1] + (secondPoint[1] - firstPoint[1])/2,
        ]    

for folder in range(1, 10):
    inputPath = PHOTO_INPUT_PATH + str(folder) + '/'
    outputPath = PHOTO_OUTPUT_PATH + str(folder) + '/'
 
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)

    dirSize = len(os.listdir((inputPath + 'data/')))
    
    for index in range(1, dirSize + 1):
        jsonFilename = inputPath + 'data/' + str(index) + '.json'
        data = json.load(open(jsonFilename))
        image = cv.imread(inputPath + 'imgs/' + str(index) + '.jpg')

        innerMargin = processJsonList(image, data['interior_margin_2d'])
        caruncle = processJsonList(image, data['caruncle_2d'])
        iris = processJsonList(image, data['iris_2d'])
        lookVector = list(eval(data['eye_details']['look_vec']))
        irisCenter = np.mean(iris[:, :2], axis=0).astype(int)

        # drawPointsAndLines(image, innerMargin, caruncle, iris)
        # drawLookVector(image, lookVector, irisCenter)

        centerPoint = findCenterOfImage(innerMargin, caruncle)
        croppedImage = cropImageAround(image, centerPoint, IMAGE_SIZE)

        cv.imwrite(outputPath + str(index) + '.jpg', croppedImage)
        # cv.imshow("syntheseyes_img", croppedImage)
        # cv.waitKey()
