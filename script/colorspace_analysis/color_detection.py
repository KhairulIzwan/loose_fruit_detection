from imutils import contours
from skimage import measure
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the image file")
args = vars(ap.parse_args())

# load the image
image = cv2.imread(args["image"])

# resize the original image
resized = imutils.resize(image, height = 400)
cv2.imshow("Resized", resized)
cv2.waitKey(0)

# converting from BGR to HSV color space
hsv = cv2.cvtColor(resized,cv2.COLOR_BGR2HSV)

# Range for lower red
lower_red = np.array([0,120,70])
upper_red = np.array([10,255,255])
mask1 = cv2.inRange(hsv, lower_red, upper_red)

# Range for upper range
lower_red = np.array([170,120,70])
upper_red = np.array([180,255,255])
mask2 = cv2.inRange(hsv,lower_red,upper_red)

# generate final mask
mask1= mask1+mask2
cv2.imshow("before filtering", mask1)
cv2.waitKey(0)

# perform erosions
mask1 = cv2.erode(mask1, None, iterations=2)
mask1 = cv2.dilate(mask1, None, iterations=4)
cv2.imshow("after filtering", mask1)
cv2.waitKey(0)

# perform cropping
res = cv2.bitwise_and(resized,resized, mask= mask1)
cv2.imshow('frame',resized)
cv2.waitKey(0)

cv2.imshow('mask',mask1)
cv2.waitKey(0)

cv2.imshow('res',res)
cv2.waitKey(0)

# reference
# https://stackoverflow.com/questions/41919319/how-do-i-know-the-position-of-white-areas-detected-using-opencv-in-python
# https://stackoverflow.com/questions/50913339/how-to-crop-area-from-mask-in-cv2-python

# perform labelling
labels = measure.label(mask1, neighbors=8, background=0)
mask = np.zeros(mask1.shape, dtype="uint8")

for label in np.unique(labels):
	# if this is the background label, ignore it
	if label == 0:
		continue
	# otherwise, construct the label mask and count the
	# number of pixels
	labelMask = np.zeros(mask1.shape, dtype="uint8")
	labelMask[labels == label] = 255
	numPixels = cv2.countNonZero(labelMask)

	# if the number of pixels in the component is sufficiently
	# large, then add it to our mask of "large blobs"
	if numPixels > 150:
		mask = cv2.add(mask, labelMask)
    # find the contours in the mask, then sort them from left to
# right
cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = contours.sort_contours(cnts)[0]

# loop over the contours
for (i, c) in enumerate(cnts):
	# draw the bright spot on the image
	(x, y, w, h) = cv2.boundingRect(c)
	((cX, cY), radius) = cv2.minEnclosingCircle(c)
	cv2.circle(resized, (int(cX), int(cY)), int(radius),
		(255, 255, 255), 2)
	cv2.putText(resized, "#{}".format(i + 1), (x, y - 15),
		cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

cv2.imshow("Image", resized)
cv2.waitKey(0)
