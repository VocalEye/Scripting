import os
import numpy as np
import cv2 as cv
import json
import multiprocessing

IMAGE_SIZE = (750, 750)
PHOTO_INPUT_PATH = 'D:\Dataset\Original Randomized Datasets\Dataset 4\\'
PHOTO_OUTPUT_PATH = './output/Dataset 4/'
NUMBER_PROCESSES = 10

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
        cv.circle(image, (int(landmark[0]), int(landmark[1])), 2, (0, 0, 255), -1)
        cv.polylines(image, np.array(
            [innerMargin[:, :2]], int), True, (255, 0, 0), 1)
        cv.polylines(image, np.array(
            [iris[:, :2]], int), True, (0, 0, 255), 1)

def drawLookVector(image, lookVector, eyeCenter):
    lookVector[1] = -lookVector[1]
    strangeValue = tuple(eyeCenter + ( np.array(lookVector[:2]) * 80).astype(int))
    cv.line(image, tuple(eyeCenter), strangeValue, (0, 0, 0), 3)
    cv.line(image, tuple(eyeCenter), strangeValue, (0, 255, 255), 2)
    
def processJsonList(image, jsonList):
    landmarks = [eval(s) for s in jsonList]
    return np.array([(x, image.shape[0]-y, z) for (x, y, z) in landmarks])

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

def divideChunks(list, chunkSize):
    for i in range(0, len(list), chunkSize):
        yield list[i:i + chunkSize]

def loadChunkFiles(index, chunk, imageSize):
    for filename in chunk:
        jsonFilename, imgFilename, outputFilename = filename
        data = json.load(open(jsonFilename))
        image = cv.imread(imgFilename)

        innerMargin = processJsonList(image, data['interior_margin_2d'])
        caruncle = processJsonList(image, data['caruncle_2d'])

        iris = processJsonList(image, data['iris_2d'])
        lookVector = list(eval(data['eye_details']['look_vec']))
        irisCenter = np.mean(iris[:, :2], axis=0).astype(int)

        # drawPointsAndLines(image, innerMargin, caruncle, iris)
        # drawLookVector(image, lookVector, irisCenter)

        centerPoint = findCenterOfImage(innerMargin, caruncle)
        croppedImage = cropImageAround(image, centerPoint, imageSize)

        cv.imwrite(outputFilename, croppedImage)

if __name__ == '__main__':
    inputFolder = PHOTO_INPUT_PATH
    outputFolder = PHOTO_OUTPUT_PATH
    
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
    
    accumulation = 1
    inputOutputMap = []

    print("READING FILES")
    for folder in range(1, 10):
        positionInputPath = inputFolder + str(folder)
       
        for set in os.listdir((positionInputPath)):
            imagesPath = positionInputPath + '/' + set + '/imgs/'
            dataPath = positionInputPath + '/' + set + '/data/'

            dirSize = len(os.listdir((imagesPath)))

            for index in range(1, dirSize):
                jsonFilename = dataPath + str(index) + '.json'
                imgFilename = imagesPath + str(index) + '.jpg'
                outputFilename = outputFolder + str(accumulation) + '_' + str(folder) + '.jpg'
                inputOutputMap.append((jsonFilename, imgFilename, outputFilename))
                accumulation += 1

    print("TOTAL FILES", accumulation)
    print("STARTING REWRITE")

    size = len(inputOutputMap)
    chunkSize = int(size/NUMBER_PROCESSES)

    chunks = list(divideChunks(inputOutputMap, chunkSize))

    jobs = []
    pipeList = []
    for index, chunk in enumerate(chunks):
        process = multiprocessing.Process(
            target = loadChunkFiles, 
            args = (index, chunk, IMAGE_SIZE)
        )
        jobs.append(process)
        process.start()
    
    for process in jobs:
        process.join()
