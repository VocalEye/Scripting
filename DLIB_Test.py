import numpy as np
import cv2 as cv
import dlib
import imutils
from scipy.spatial import distance as dist
from imutils import face_utils

cap = cv.VideoCapture(1)
detector = dlib.get_frontal_face_detector()
landmark_predict = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')

FACIAL_LANDMARKS_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 35)),
	("jaw", (0, 17))
])

def visualize_facial_landmarks(image, shape, colors=None, alpha=0.75):
	# create two copies of the input image -- one for the
	# overlay and one for the final output image
	overlay = image.copy()
	output = image.copy()
	# if the colors list is None, initialize it with a unique
	# color for each facial landmark region
	if colors is None:
		colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
			(168, 100, 168), (158, 163, 32),
			(163, 38, 32), (180, 42, 220)]
    
    # loop over the facial landmark regions individually
	for (i, name) in enumerate(FACIAL_LANDMARKS_IDXS.keys()):
		# grab the (x, y)-coordinates associated with the
		# face landmark
		(j, k) = FACIAL_LANDMARKS_IDXS[name]
		pts = shape[j:k]
		# check if are supposed to draw the jawline
		if name == "jaw":
			# since the jawline is a non-enclosed facial region,
			# just draw lines between the (x, y)-coordinates
			for l in range(1, len(pts)):
				ptA = tuple(pts[l - 1])
				ptB = tuple(pts[l])
				cv.line(overlay, ptA, ptB, colors[i], 2)
		# otherwise, compute the convex hull of the facial
		# landmark coordinates points and display it
		else:
			hull = cv.convexHull(pts)
			cv.drawContours(overlay, [hull], -1, colors[i], -1)
	
    # apply the transparent overlay
	cv.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
	# return the output image
	return output

if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    ret, frame = cap.read()
    
    if not ret or np.sum(frame) == 0:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()