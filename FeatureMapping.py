import os
import numpy as np
import cv2 as cv
import json
import multiprocessing

IMAGE_SIZE = (750, 750)
PHOTO_INPUT_PATH = 'D:\Dataset\Original Randomized Datasets\Dataset 4\\'
PHOTO_OUTPUT_PATH = './output/Dataset 4 Features/'
# PHOTO_INPUT_PATH = 'D:\Dataset\Real Datasets\Real Dataset 2\\'
# PHOTO_OUTPUT_PATH = './output/Real Dataset 2/'
NUMBER_PROCESSES = 20

def processJsonList(image, jsonList):
    landmarks = [eval(s) for s in jsonList]
    return np.array([(x, image.shape[0]-y, z) for (x, y, z) in landmarks])

def findMarginOfEye(eyeCenter, imageSize):
    widthHalf = imageSize[0] / 2
    heightHalf = imageSize[1] / 2
    firstPoint = (int(eyeCenter[0] - widthHalf), int(eyeCenter[1] - heightHalf))
    secondPoint = (int(eyeCenter[0] + widthHalf), int(eyeCenter[1] + heightHalf))

    return firstPoint, secondPoint

def cropImageAround(image, firstPoint, secondPoint):
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
        centerPoint = findCenterOfImage(innerMargin, caruncle)
        firstPoint, secondPoint = findMarginOfEye(centerPoint, imageSize)
        
        merge = np.append(
            innerMargin[:, :2].astype(int), 
            caruncle[:, :2].astype(int), 
            axis=0
        )
        hull = cv.convexHull(merge)
        
        featureMap = printFeatureMap(imageSize, firstPoint, hull, iris)
        cv.imwrite(outputFilename, featureMap)

def create_blank(width, height, rgb_color=(0, 0, 0)):
    image = np.zeros((height, width, 3), np.uint8)

    color = tuple(reversed(rgb_color))

    image[:] = color
    return image

def printFeatureMap(imageSize, firstPoint, meshEye, meshIris):
    white = (255, 255, 255)
    imageBlank = create_blank(imageSize[0], imageSize[1], rgb_color = white)
    
    movedMeshIris = np.array([[
            int(point[0] - firstPoint[0]),
            int(point[1] - firstPoint[1]),
        ] for point in meshIris[:, :2]])
    cv.fillPoly(imageBlank, [movedMeshIris], (0, 0, 255))

    movedMeshEye = np.array([[
            int(point[0][0] - firstPoint[0]),
            int(point[0][1] - firstPoint[1]),
        ] for point in meshEye])
    
    mask = np.zeros(imageBlank.shape[0:2], dtype=np.uint8)
    cv.drawContours(mask, [movedMeshEye], -1, (255, 255, 255), -1, cv.LINE_AA)
    imageBlank = cv.bitwise_and(imageBlank, imageBlank, mask = mask)

    return imageBlank

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
