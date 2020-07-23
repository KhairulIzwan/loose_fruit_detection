#!/usr/bin/env python3

# Title: Segmentation of Oil Palm Loose Fruit using Colorspace Technique
# Description: Using different type of colorspace -- RGB and HSV -- to segmenting the loose fruit -- masking
# Author: Universiti Putra Malaysia (FYP)

# import useful library
import numpy as np
import argparse
import cv2

import imutils

# parsing the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
# ap.add_argument("-r", "--radius", type=int, required=True, help="Radius of Gaussian blur; must be odd")
args = vars(ap.parse_args())

# STEP 1: ================== Load an Image ==================
# read an image
image = cv2.imread(args["image"])

# STEP 2: ================== Resize ==================
# to reduce processing time
resized = imutils.resize(image, width=image.shape[1]//2, inter=3)

# STEP 3: ================== ColorSpace ==================
# convert image to grayscale
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
# convert image to hsv
hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

# STEP 4: ================== Smoothing and Blurring ==================
blurred_gray = cv2.GaussianBlur(gray, (5, 5), 0)
blurred_hsv = cv2.GaussianBlur(hsv, (5, 5), 0)

# STEP 5: ================== Thresholding ==================
(T_gray, thresh_gray) = cv2.threshold(blurred_gray, 155, 255, cv2.THRESH_BINARY)

# STEP 6: ================== Masking ==================
mask_gray = cv2.erode(thresh_gray, None, iterations=2)
mask_gray = cv2.dilate(mask_gray, None, iterations=2)

# STEP (OPTIONAL): ================== Display ==================
cv2.imshow("Original Image", resized)
cv2.imshow("Image Processing (Grayscale)", np.hstack([gray, blurred_gray, thresh_gray, mask_gray]))

cv2.waitKey(0)
