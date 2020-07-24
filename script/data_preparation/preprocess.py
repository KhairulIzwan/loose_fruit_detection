#!/usr/bin/env python3

# import useful library
import numpy as np
import argparse
import cv2
from imutils import paths
import imutils
from skimage import measure
from imutils import contours
import os

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
ap.add_argument("-i", "--input", required=True, 
	help="path to input directory of images")
ap.add_argument("-a", "--annot", required=True, 
	help="path to output directory of annotations")
ap.add_argument("-f", "--folder", required=True,
	help="folder name")
args = vars(ap.parse_args())

# grab the image paths then initialize the dictionary of character
# counts
imagePaths = list(paths.list_images(args["input"]))
count = 0
countD = 0

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
	# display an update to the user
	print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))

	try:
		while True:
			# load the image
			image = cv2.imread(imagePath)
			imageClone = image.copy()

			# convert to hsv colorspace
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

			# resize image for display purpose only
			resized_img = imutils.resize(image, 
				width=image.shape[1]//3, inter=3)
			resized_result = imutils.resize(result, 
				width=image.shape[1]//3, inter=3)

			cv2.imshow("Color Segmentation", np.hstack([resized_img, resized_result]))
				
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

				# if the number of pixels in the component is
				# sufficiently large, then add it to our mask of
				# "large blobs"
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
			
				# offset
				x -= 5
				y -= 5
				w += 10
				h += 10
			
				cv2.rectangle(image, (int(x), int(y)), 
					(int(x+w), int(y+h)), (255, 255, 255), 2)
				# ((cX, cY), radius) = cv2.minEnclosingCircle(c)
				# cv2.circle(image, (int(cX), int(cY)), int(radius),
				#	(255, 255, 255), 2)
				cv2.putText(image, "#{}".format(i + 1), (x, y - 15),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255,
					 255), 2)
			
				roi = imageClone[y:y + h, x:x + w]
				cv2.imshow("ROI", roi)
				
				# construct the path the output directory
				dirPath = os.path.join(args["annot"], args["folder"])

				# if the output directory does not exist, create it
				if not os.path.exists(dirPath):
					os.makedirs(dirPath)

				# write the labeled character to file
				p = os.path.sep.join([dirPath, "{}.png".format(
					str(countD).zfill(6))])
				cv2.imwrite(p, roi)
				
				# increment the count for the current key
				countD = countD + 1
				
			key = cv2.waitKey(0)
			# if the 'q' key is pressed, break from the loop
			if key == ord("q"):
				break

			# increment the count for the current key
			count = count + 1

	# we are trying to control-c out of the script, so break from the
	# loop (you still need to press a key for the active window to
	# trigger this)
	except KeyboardInterrupt:
		print("[INFO] manually leaving script")
		break

	# an unknown error has occurred for this particular image
	except:
		print("[INFO] skipping image...")

# cap.release()
cv2.destroyAllWindows()
