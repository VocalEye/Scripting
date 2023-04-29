import cv2 as cv
import os

IMAGE_SIZE = (800, 800)
DISPLACEMENT = (150, 200)
FACTOR = 2
CLASS = 8
PRINT = 0
PHOTO_INPUT_PATH = '../Test/' + str(CLASS) + "/"
PHOTO_OUTPUT_PATH = './output/'

def cropImageAround(image, size, factor, displacement):
    imageCrop = image[
        displacement[1] : (size[1]*factor) + displacement[1], 
        displacement[0] : (size[0]*factor) + displacement[0]
        ]
    return cv.resize(imageCrop, size)

accumulator = len(os.listdir(PHOTO_OUTPUT_PATH)) + 1
for filename in os.listdir(PHOTO_INPUT_PATH):
    image = cv.imread(PHOTO_INPUT_PATH + filename)

    croppedImage = cropImageAround(image, IMAGE_SIZE, FACTOR, DISPLACEMENT)

    if PRINT:
        cv.imshow("syntheseyes_img", croppedImage)
        cv.waitKey()
        break
    
    cv.imwrite(PHOTO_OUTPUT_PATH + str(accumulator) + '_' + str(CLASS + 1) + '.jpg', croppedImage)
    accumulator += 1
