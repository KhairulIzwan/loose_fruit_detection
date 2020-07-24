#!/usr/bin/env python3

# Title: Segmentation of Oil Palm Loose Fruit using Colorspace Technique
# Description: Using different type of colorspace -- RGB and HSV -- to segmenting the loose fruit -- masking
# Author: Universiti Putra Malaysia (FYP)

"""
We have an RGB (Red-Green-Blue) image and it is tempting to simply threshold the R channel and get our mask. It turns out that this will not work effectively since the RGB values are highly sensitive to illumination. Hence even though the object is of red color there might be some areas where, due-to shadow, Red channel values of the corresponding pixels are quite low.

The right approach is to transform the color space of our image from RGB to HSV (Hue-Saturation-Value).

HSV?
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

# import useful library
import numpy as np
import argparse
import cv2

import imutils
from skimage import measure
from imutils import contours

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
#image = imutils.resize(image, width=image.shape[1]//2, inter=3)

# STEP 2: ================== ColorSpace ==================
# convert image to hsv
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# STEP 3: ================== Masking ==================
# Range for lower red
lower_red = np.array([0,120,70])
upper_red = np.array([10,255,255])
# mask1 = cv2.inRange(blurred, lower_red, upper_red)
mask1 = cv2.inRange(hsv, lower_red, upper_red)

# Range for upper range
lower_red = np.array([170,120,70])
upper_red = np.array([180,255,255])
# mask2 = cv2.inRange(blurred,lower_red, upper_red)
mask2 = cv2.inRange(hsv,lower_red, upper_red)

# Generating the final mask to detect red color
mask = mask1 + mask2

mask = cv2.erode(mask, None, iterations=1)
mask = cv2.dilate(mask, None, iterations=1)

#Segmenting the cloth out of the frame using bitwise and with the inverted mask
res1 = cv2.bitwise_and(image, image, mask=mask)

# STEP 4: ================== Counting and Labelling ==================
# perform labelling
labels = measure.label(mask, neighbors=8, background=0)
mask = np.zeros(mask.shape, dtype="uint8")

for label in np.unique(labels):
	# if this is the background label, ignore it
	if label == 0:
		continue
	# otherwise, construct the label mask and count the
	# number of pixels
	labelMask = np.zeros(mask.shape, dtype="uint8")
	labelMask[labels == label] = 255
	numPixels = cv2.countNonZero(labelMask)

	# if the number of pixels in the component is sufficiently
	# large, then add it to our mask of "large blobs"
	if numPixels > 100:
		mask = cv2.add(mask, labelMask)
    # find the contours in the mask, then sort them from left to
# right
cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = contours.sort_contours(cnts)[0]

print("I count {} fruits (maybe not!) in this image".format(len(cnts)))

# STEP X: ================== Display ==================
# loop over the contours
for (i, c) in enumerate(cnts):
	# draw the bright spot on the image
	(x, y, w, h) = cv2.boundingRect(c)
	((cX, cY), radius) = cv2.minEnclosingCircle(c)

	fruit = image[y:y + h, x:x + w]
	cv2.imshow("Fruit", fruit)

	cv2.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)), (255, 255, 255), 2)
	cv2.circle(image, (int(cX), int(cY)), int(radius), (255, 255, 255), 2)
	cv2.putText(image, "#{}".format(i + 1), (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

	cv2.imshow("HSV Masking", np.hstack([image, res1]))
	cv2.waitKey(0)
