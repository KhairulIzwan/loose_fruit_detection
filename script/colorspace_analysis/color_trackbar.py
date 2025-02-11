#!/usr/bin/env python

# import useful library
import numpy as np
import argparse
import cv2

import imutils

def nothing(x):
    pass

# cap = cv2.VideoCapture(0)
cv2.namedWindow("Trackbars")

cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

# parsing the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load an image
image = cv2.imread(args["image"])

while True:
	# _, frame = cap.read()
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

	l_h = cv2.getTrackbarPos("L - H", "Trackbars")
	l_s = cv2.getTrackbarPos("L - S", "Trackbars")
	l_v = cv2.getTrackbarPos("L - V", "Trackbars")
	u_h = cv2.getTrackbarPos("U - H", "Trackbars")
	u_s = cv2.getTrackbarPos("U - S", "Trackbars")
	u_v = cv2.getTrackbarPos("U - V", "Trackbars")

	lower_blue = np.array([l_h, l_s, l_v])
	upper_blue = np.array([u_h, u_s, u_v])
	mask = cv2.inRange(hsv, lower_blue, upper_blue)

	result = cv2.bitwise_and(image, image, mask=mask)

	# resize image
	resized_image = imutils.resize(image, width=image.shape[1]//2, inter=3)
	resized_result = imutils.resize(result, width=image.shape[1]//2, inter=3)

	cv2.imshow("Original --> Colorspace", np.hstack([resized_image, resized_result]))

	key = cv2.waitKey(1)
	if key == 27:
		break

# cap.release()
cv2.destroyAllWindows()
