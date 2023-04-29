import os
import numpy as np
import cv2 as cv
import json
import multiprocessing

IMAGE_SIZE = (750, 750)
PHOTO_INPUT_PATH = '../Original/Dataset '
PHOTO_OUTPUT_PATH = './output/Dataset '
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

        drawPointsAndLines(image, innerMargin, caruncle, iris)
        drawLookVector(image, lookVector, irisCenter)

        centerPoint = findCenterOfImage(innerMargin, caruncle)
        croppedImage = cropImageAround(image, centerPoint, imageSize)

        cv.imwrite(outputFilename, croppedImage)

if __name__ == '__main__':
    for main in range(1, 4):
        inputFolder = PHOTO_INPUT_PATH + str(main) + '/'
        outputTrainFolder = PHOTO_OUTPUT_PATH + str(main) + '/train/'
        outputTestFolder = PHOTO_OUTPUT_PATH + str(main) + '/test/'
        
        if not os.path.exists(outputTrainFolder):
            os.makedirs(outputTrainFolder)
        if not os.path.exists(outputTestFolder):
            os.makedirs(outputTestFolder)
        
        accumulationTrain = 1
        accumulationTest = 1
        inputOutputMap = []

        for folder in range(1, 10):
            imageInputPath = inputFolder + str(folder) + '/imgs/'
            dataInputPath = inputFolder + str(folder) + '/data/'
            dirSize = len(os.listdir((imageInputPath)))
            
            for index in range(1, 4501):
                jsonFilename = dataInputPath + str(index) + '.json'
                imgFilename = imageInputPath + str(index) + '.jpg'
                outputFilename = outputTrainFolder + str(accumulationTrain) + '_' + str(folder) + '.jpg'
                inputOutputMap.append((jsonFilename, imgFilename, outputFilename))
                accumulationTrain += 1
            
            for index in range(4501, dirSize + 1):
                jsonFilename = dataInputPath + str(index) + '.json'
                imgFilename = imageInputPath + str(index) + '.jpg'
                outputFilename = outputTestFolder + str(accumulationTest) + '_' + str(folder) + '.jpg'
                inputOutputMap.append((jsonFilename, imgFilename, outputFilename))
                accumulationTest += 1
    
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
    
                
            
