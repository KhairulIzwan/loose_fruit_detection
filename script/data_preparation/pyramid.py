#!/usr/bin/env python3

# Title: Image pyramid
# Description: An operation of building image pyramid
# Author: Tutorial from Practical Python and OpenCV, 4th Edition.pdf

# import useful library
from skimage.transform import pyramid_gaussian

# TODO:
# bring your package onto the path
import sys, os
sys.path.append(os.path.abspath(os.path.join("../")))

# Now do your import
from library.helpers import pyramid

import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
ap.add_argument("-s", "--scale", type=float, default=1.5, help="scale factor size")
args = vars(ap.parse_args())

# load the image
image = cv2.imread(args["image"])

# METHOD #1: No smooth, just scaling.
# loop over the image pyramid
for (i, resized) in enumerate(pyramid(image, scale=args["scale"])):
    # show the resized image
    cv2.imshow("Layer {}".format(i + 1), resized)
    cv2.waitKey(0)

# close all windows
cv2.destroyAllWindows()

# METHOD #2: Resizing + Gaussian smoothing.
for (i, resized) in enumerate(pyramid_gaussian(image, downscale=2)):
    # if the image is too small, break from the loop
    if resized.shape[0] < 30 or resized.shape[1] < 30:
        break

    # show the resized image
    cv2.imshow("Layer {}".format(i + 1), resized)
    cv2.waitKey(0)
