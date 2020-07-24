#!/usr/bin/env python3

# Title: ColorSpace in OpenCV
# Description: Re-visit color space operation OpenCV
# Author: Tutorial from Practical Python and OpenCV, 4th Edition.pdf

# import useful library
import numpy as np
import imutils
import argparse
import cv2

# parsing the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load an image
image = cv2.imread(args["image"])

# resize image
#resized = imutils.resize(image, width=image.shape[1]//2, inter=3)

# Colorspace in OpenCV
resized_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
resized_hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

# display image (OPTIONAL) -- uncomment to view
cv2.imshow("ColorSpace (BGR, GRAY, HSV)", np.hstack([resized, cv2.merge([resized_gray, resized_gray, resized_gray]), resized_hsv]))

# wait for key pressed
cv2.waitKey(0)
