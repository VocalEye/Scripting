import os
import shutil

PHOTO_INPUT_PATH = '../Randomized Camera Dataset #3/'
PHOTO_OUTPUT_PATH = './output-2/'

for folder in range(1, 10):
    inputPath = PHOTO_INPUT_PATH + str(folder) + '/'
    outputPath = PHOTO_OUTPUT_PATH + str(folder) + '/'
 
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
        os.makedirs(outputPath + 'data')
        os.makedirs(outputPath + 'imgs')

    accumulation = 1
    for position in range(1, 7):
        positionPath = inputPath + 'Position ' + str(position) + '/'
        dirSize = 834 # len(os.listdir((positionPath + 'imgs/')))
        
        for index in range(1, dirSize + 1):
            shutil.copy(
                positionPath + 'imgs/' + str(index) + '.jpg',
                outputPath + 'imgs/' + str(accumulation) + '.jpg'
                )
            shutil.copy(
                positionPath + 'data/' + str(index) + '.json',
                outputPath + 'data/' + str(accumulation) + '.json'
                )
            accumulation += 1
            
