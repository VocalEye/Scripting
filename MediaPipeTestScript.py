import cv2
import mediapipe as mp
import numpy as np
import os

mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE = [362,382,381,380,374,390,249,263,466,388,387,386,385,384,398]
RIGHT_EYE = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
LEFT_IRIS = [474,475,476,477]
RIGHT_IRIS = [469,470,471,472]
FROM_WEBCAM = False
MARGIN = 30

INPUT = 'D:\Dataset\Real Datasets\Real Dataset 3\\'
OUTPUT = './output-2/'

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

def paintReference(image, mesh_points):
  cv2.polylines(image, [mesh_points[LEFT_EYE]], True, (0, 255,0), 1, cv2.LINE_AA)
  cv2.polylines(image, [mesh_points[RIGHT_EYE]], True, (0, 255,0), 1, cv2.LINE_AA)

  (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
  (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])
  center_left = np.array([l_cx, l_cy], dtype=np.int32)
  center_right = np.array([r_cx, r_cy], dtype=np.int32)

  cv2.circle(image, center_left, int(l_radius), (255,0,255), 1, cv2.LINE_AA)
  cv2.circle(image, center_right, int(r_radius), (255,0,255), 1, cv2.LINE_AA)

def create_blank(width, height, rgb_color=(0, 0, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((height, width, 3), np.uint8)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color

    return image

def printFeatureMap(imageSize, firstPoint, meshEye, meshIris):
    white = (255, 255, 255)
    imageBlank = create_blank(imageSize[0], imageSize[1], rgb_color = white)
    movedMeshEye = np.array([[
            point[0] - firstPoint[0],
            point[1] - firstPoint[1],
        ] for point in meshEye])
    (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(meshIris)
    center_right = np.array([r_cx - firstPoint[0], r_cy - firstPoint[1]], dtype=np.int32)

    # cv2.circle(imageBlank, (movedMeshEye[0][0], movedMeshEye[0][1]), 3, (255,0,255), -1)
    cv2.circle(imageBlank, center_right, int(r_radius), (0,0,255), -1, cv2.LINE_AA)

    mask = np.zeros(imageBlank.shape[0:2], dtype=np.uint8)
    cv2.drawContours(mask, [movedMeshEye], -1, (255, 255, 255), -1, cv2.LINE_AA)
    imageBlank = cv2.bitwise_and(imageBlank, imageBlank, mask = mask)

    return cv2.flip(imageBlank, 1)

def readFromWebcam(face_mesh):
    capturer = getCapturer()
    while capturer.isOpened():
        success, image = capturer.read()
        if not success:
            print("Ignoring empty camera frame.")
            break

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)
        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        imageHeight, imageWeight = image.shape[:2]
        if results.multi_face_landmarks:
            mesh_points = np.array([ np.multiply([p.x, p.y], [imageWeight, imageHeight]).astype(int) for p in results.multi_face_landmarks[0].landmark])

            paintReference(image, mesh_points)
            # 33 - 133
            # cv2.circle(image, (int(mesh_points[33][0]), int(mesh_points[33][1])), 3, (255, 0, 0), -1)
            # cv2.circle(image, (int(mesh_points[133][0]), int(mesh_points[133][1])), 3, (255, 0, 0), -1)
            # mid_x = (mesh_points[33][0] + mesh_points[133][0])/2
            # mid_y = (mesh_points[33][1] + mesh_points[133][1])/2
            # cv2.circle(image, (int(mid_x), int(mid_y)), 3, (255, 0, 0), -1)

            # radius = (((mesh_points[33][0] - mesh_points[133][0])**2 + (mesh_points[33][1] - mesh_points[133][1])**2)**0.5)/2
            #

            #cv2.circle(image, (int(mid_x), int(mid_y)), int(radius), (255,0,255), 1, cv2.LINE_AA)
            firstPoint, secondPoint = findMarginOfEye(mesh_points[RIGHT_EYE])
            image = cropImageAround(image, firstPoint, secondPoint)
            
            features = printFeatureMap((len(image), len(image[0])), firstPoint, mesh_points[RIGHT_EYE], mesh_points[RIGHT_IRIS])
            cv2.imshow('MediaPipe Face Mesh', features)

        processImage(face_mesh, image)
        cv2.imshow('MediaPipe Face Mesh', image)
        key = cv2.waitKey(1) 
        if key == ord('q'):
            break
    capturer.release()

def readFromFiles(face_mesh):
    files = os.listdir(INPUT)
    for file in files:
        image = cv2.imread(INPUT + file)
        number, quadrant = map(int, file[:-4].split('_'))
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)
        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        imageHeight, imageWeight = image.shape[:2]
        if results.multi_face_landmarks:
            mesh_points = np.array([ np.multiply([p.x, p.y], [imageWeight, imageHeight]).astype(int) for p in results.multi_face_landmarks[0].landmark])

            # firstPoint, secondPoint = findMarginOfEye(mesh_points[RIGHT_EYE])
            # print(firstPoint, secondPoint)
            # # cv2.rectangle(image, firstPoint, secondPoint, (255, 0, 0), 2)
            # image = cropImageAround(image, firstPoint, secondPoint)

            firstPoint, secondPoint = findMarginOfEye(mesh_points[RIGHT_EYE])
            image = cropImageAround(image, firstPoint, secondPoint)
            image = printFeatureMap((len(image), len(image[0])), firstPoint, mesh_points[RIGHT_EYE], mesh_points[RIGHT_IRIS])
            cv2.imwrite(OUTPUT + str(number) + '_' + str(quadrant) + '.jpg', image)

        # image = cv2.flip(image, 1)

        # cv2.imshow('MediaPipe Face Mesh', image)
        # cv2.imwrite(OUTPUT + str(number) + '_' + str(quadrant) + '.jpg', image)
        key = cv2.waitKey(1) 
        if key == ord('q'):
            continue

if __name__ == '__main__':
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.95,
        min_tracking_confidence=0.999) as face_mesh:
        if FROM_WEBCAM:
            readFromWebcam(face_mesh)        
        else:
            readFromFiles(face_mesh)