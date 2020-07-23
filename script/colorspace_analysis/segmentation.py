#!/usr/bin/env python3

# import useful library
import numpy as np
import argparse
import cv2

import imutils

import matplotlib.pyplot as plt

# parsing the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load an image
image = cv2.imread(args["image"])

# resize image
resized = imutils.resize(image, width=image.shape[1]//2, inter=3)

# BGR (OpenCV) --> RGB (Matplotlib)
resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
resized_hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
resized_hsv2 = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2HSV)

# h, s, v = cv2.split(resized_hsv)

# display image (OPTIONAL) -- uncomment to view
cv2.imshow("Original Image", np.hstack([resized_rgb, resized_hsv, resized_hsv2]))
# cv2.imshow("Original Image (HSV)", np.hstack([h, s, v]))

#
fig = plt.figure()

ax = fig.add_subplot(121)
p = ax.imshow(resized, interpolation="nearest")
ax.set_title("RGB ColorSpace")

# TODO:
# plt.colorbar(p)

ax = fig.add_subplot(122)
p = ax.imshow(resized_hsv, interpolation="nearest")
ax.set_title("HSV ColorSpace")

# TODO:
# plt.colorbar(p)

plt.show()

# wait for key pressed
cv2.waitKey(0)
