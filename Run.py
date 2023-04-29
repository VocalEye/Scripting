import cv2
import mediapipe as mp
import numpy as np
import os
from keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE = [362,382,381,380,374,390,249,263,466,388,387,386,385,384,398]
RIGHT_EYE = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
LEFT_IRIS = [474,475,476,477]
RIGHT_IRIS = [469,470,471,472]
FROM_WEBCAM = True
MARGIN = 30
MODEL = load_model('./models/model.h5')

INPUT = './output/Real Dataset 4/'
OUTPUT = './output-2/'

def getPredictedClass(predictions):
    y_pred_class = []

    for prediction in predictions:
        search = np.where(prediction == np.amax(prediction))[0]
        if not search.size:
            return None
        y_pred_class.append(search[0])
    
    return y_pred_class

def getCapturer():
    capture = False
    while(not capture):
        capture = cv2.VideoCapture(1)
        if not capture.isOpened():
            print("Cannot open camera")
            exit()

        ret, frame = capture.read()
        if not ret or np.sum(frame) == 0:
            print("Can't receive frame (stream end?). Retrying...")
            capture.release()

            capture = cv2.VideoCapture(0)
            capture.release()
            capture = False
    return capture

def cropImageAround(image, firstPoint, secondPoint):
    return image[firstPoint[1]:secondPoint[1], firstPoint[0]:secondPoint[0]]

def findMarginOfEye(innerMargin):
    innerMarginX = list(map(lambda margin: margin[0], innerMargin))
    innerMarginY = list(map(lambda margin: margin[1], innerMargin))

    innerMarginX.sort()
    innerMarginY.sort()

    firstPoint = (innerMarginX[0] - MARGIN, innerMarginY[0] - MARGIN)
    secondPoint = (innerMarginX[len(innerMarginX)-1] + MARGIN, innerMarginY[len(innerMarginY)-1] + MARGIN)
    horizontal = (secondPoint[0] - firstPoint[0])
    vertical = (secondPoint[1] - firstPoint[1])
    difference = int((horizontal-vertical)/2)

    return (firstPoint[0], firstPoint[1] - difference), (secondPoint[0], secondPoint[1] + difference)

def processImage(faceMesh, image):
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(image)
    
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    imageHeight, imageWeight = image.shape[:2]
    if results.multi_face_landmarks:
        mesh_points = np.array([ np.multiply([p.x, p.y], [imageWeight, imageHeight]).astype(int) for p in results.multi_face_landmarks[0].landmark])

        firstPoint, secondPoint = findMarginOfEye(mesh_points[RIGHT_EYE])
        # cv2.rectangle(image, firstPoint, secondPoint, (255, 0, 0), 2)
        image = cropImageAround(image, firstPoint, secondPoint)
    image = cv2.flip(image, 1)
    
    return image

def CustomParser(image):    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (32, 32))
    resized = resized[np.newaxis,:,:]
    return resized/255

if __name__ == '__main__':
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.5) as face_mesh:
        
        capturer = getCapturer()
        while capturer.isOpened():
            success, full_image = capturer.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                break

            full_image.flags.writeable = False
            full_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(full_image)
            
            full_image.flags.writeable = True
            full_image = cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR)
            imageHeight, imageWeight = full_image.shape[:2]
            if results.multi_face_landmarks:
                mesh_points = np.array([ np.multiply([p.x, p.y], [imageWeight, imageHeight]).astype(int) for p in results.multi_face_landmarks[0].landmark])

                firstPoint, secondPoint = findMarginOfEye(mesh_points[RIGHT_EYE])
                
                full_image = cropImageAround(full_image, firstPoint, secondPoint)
            full_image = cv2.flip(full_image, 1)
            
            # processImage(face_mesh, image)
            cv2.imshow('MediaPipe Face Mesh', full_image)
             
            parsed = CustomParser(full_image)
            y_pred = getPredictedClass(MODEL.predict(parsed, verbose = 0))
            print(str(y_pred[0]+1) , end='\r')

            key = cv2.waitKey(1) 
            if key == ord('q'):
                break
        capturer.release()

