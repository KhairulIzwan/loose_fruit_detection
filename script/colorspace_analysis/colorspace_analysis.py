#!/usr/bin/env python3

"""
Title: Color space analysis
Description: Implementing different type colorspace - RGB, GRAY, HSV, etc - on subject oil palm loose fruit for color segmentation purpose
Author: Universiti Putra Malaysia (FYP)
"""

"""
NOTES: Dataset were collected using iPhone (handheld)
"""

# import useful library
import numpy as np
import argparse
import cv2

import imutils

# parsing the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load an image
image = cv2.imread(args["image"])

# TODO: ================== Resize ==================
# print the image data: shape (OPTIONAL) -- width, height and channels -- uncomment to view
# print("width: %d pixels" % (image.shape[1]))
# print("height: %d pixels" % (image.shape[0]))
# print("channels: %d pixels" % (image.shape[2]))

# display image (OPTIONAL) -- uncomment to view
# cv2.imshow("Image", image)

# resize image
resized = imutils.resize(image, width=image.shape[1]//2, inter=3)

# print the image data: shape (OPTIONAL) -- width, height and channels -- uncomment to view
# print("width (NEW): %d pixels" % (resized.shape[1]))
# print("height (NEW): %d pixels" % (resized.shape[0]))
# print("channels (NEW): %d pixels" % (resized.shape[2]))

# display image (OPTIONAL) -- uncomment to view
# cv2.imshow("Resized", image)

# TODO: ================== ColorSpace ==================
# # convert image to grayscale
# orig = image.copy()
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# # display image
# cv2.imshow("Grayscale", gray)

# TODO: Added to replace grayscale colorspace converter
"""
We have an RGB (Red-Green-Blue) image and it is tempting to simply threshold the R channel and get our mask. It turns out that this will not work effectively since the RGB values are highly sensitive to illumination. Hence even though the object is of red color there might be some areas where, due-to shadow, Red channel values of the corresponding pixels are quite low.
The right approach is to transform the color space of our image from RGB to HSV (Hue-Saturation-Value).

HSV 
1. Hue : This channel encodes color color information. Hue can be thought of an angle where 0 degree corresponds to the red color, 120 degrees corresponds to the green color, and 240 degrees corresponds to the blue color.
2. Saturation : This channel encodes the intensity/purity of color. For example, pink is less saturated than red.
3. Value : This channel encodes the brightness of color. Shading and gloss components of an image appear in this channel.

Unlike RGB which is defined in relation to primary colors, HSV is defined in a way that is similar to how humans perceive color.

Reference:
1. https://www.learnopencv.com/invisibility-cloak-using-color-detection-and-segmentation-with-opencv/
2. https://realpython.com/python-opencv-color-spaces/
3. https://www.learnopencv.com/color-spaces-in-opencv-cpp-python/
4. https://www.learnopencv.com/color-spaces-in-opencv-cpp-python/
"""

# converting from BGR to HSV color space
resized_orig = resized.copy()
hsv = cv2.cvtColor(resized_orig, cv2.COLOR_BGR2HSV)

# display image (OPTIONAL) -- uncomment to view
# cv2.imshow("HSV", hsv)

cv2.imshow("Image Processing -- (ColorSpace)", np.hstack([resized_orig, hsv]))

# # susceptible method: find brightest spot without image processing
# (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
# cv2.circle(image, maxLoc, 5, (255, 0, 0), 2)
#
# # display image
# cv2.imshow("Naive", image)
#
# # robust method: find brightest spot with image processing
#
# # apply gaussian blur to the image, then find the brightest region
# gray = cv2.GaussianBlur(gray, (args["radius"], args["radius"]), 0)
#
# # display image
# cv2.imshow("Blur", gray)
#
# (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
# image = orig.copy()
# cv2.circle(image, maxLoc, 5, (255, 0, 0), 2)
#
# # display image
# cv2.imshow("Robust", image)

cv2.waitKey(0)
