#!/usr/bin/env python

#Title: Python Subscriber for Tank Navigation
#Author: Khairul Izwan Bin Kamsani - [23-01-2020]
#Description: Tank Navigation Subcriber Nodes (Python)

from __future__ import print_function
from __future__ import division

# import the necessary packages
from imutils import face_utils
from collections import deque
import imutils
import time
import cv2
import os
import rospkg
import sys
import rospy
import numpy as np

# import the necessary ROS messages
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo

from cv_bridge import CvBridge
from cv_bridge import CvBridgeError

from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage

class ColorDetection:

	def __init__(self, buffer=16):

		rospy.logwarn("[Robot1] Color Detection node [ONLINE]")

		self.bridge = CvBridge()

		# define the lower and upper boundaries of the "oil palm"
		# in the HSV color space, then initialize the
		# list of tracked points
		self.lower_red = (0, 120, 70)
        	self.upper_red = (10, 255, 255)
        	self.lowerRed = (170, 120, 70)
        	self.upperRed = (180, 255, 255)

		self.pts = deque(maxlen=buffer)
		self.buffer = buffer

		self.image_recieved = False

		# rospy shutdown
		rospy.on_shutdown(self.cbShutdown)

		# Subscribe to Image msg
		image_topic = "/cv_camera_robot1/image_raw"
		self.image_sub = rospy.Subscriber(image_topic, Image, self.cbImage)

		# Subscribe to CameraInfo msg
		cameraInfo_topic = "/cv_camera_robot1/camera_info"
		self.cameraInfo_sub = rospy.Subscriber(cameraInfo_topic, CameraInfo,
			self.cbCameraInfo)

		# Allow up to one second to connection
		rospy.sleep(1)

	# Convert image to OpenCV format
	def cbImage(self, msg):

		try:
			self.cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
			self.cv_image = cv2.flip(self.cv_image, 1)
		except CvBridgeError as e:
			print(e)

		if self.cv_image is not None:
			self.image_received = True
		else:
			self.image_received = False

	# Get CameraInfo
	def cbCameraInfo(self, msg):

		self.imgWidth = msg.width
		self.imgHeight = msg.height

		# calculate the center of the frame as this is where we will
		# try to keep the object
		self.centerX = self.imgWidth // 2
		self.centerY = self.imgHeight // 2

	# Show the output frame
	def cbShowImage(self):

		cv2.imshow("[Robot1] Loose Fruit Detector (ROI)", self.cv_image)
		cv2.waitKey(1)

	# Image information callback
	def cbInfo(self):

		fontFace = cv2.FONT_HERSHEY_SIMPLEX
		fontScale = 0.4
		color = (255, 255, 255)
		thickness = 1
		lineType = cv2.LINE_AA
		bottomLeftOrigin = False # if True (text upside down)

#		self.timestr = time.strftime("%Y%m%d-%H:%M:%S")

#		cv2.putText(self.cv_image, "{}".format(self.timestr), (10, 20), 
#			fontFace, fontScale, color, thickness, lineType, 
#			bottomLeftOrigin)
		cv2.putText(self.cv_image, "Possible Fruit: %d"% (len(np.unique(self.labels)) - 1), (5, 20), 
			fontFace, fontScale, (0, 0, 255), thickness, lineType, 
			bottomLeftOrigin)	
#		cv2.putText(self.cv_image, "Sample", (10, self.imgHeight-10), 
#			fontFace, fontScale, color, thickness, lineType, 
#			bottomLeftOrigin)
#		cv2.putText(self.cv_image, "(%d, %d)" % (self.imgWidth, self.imgHeight), 
#			(self.imgWidth-100, self.imgHeight-10), fontFace, fontScale, 
#			color, thickness, lineType, bottomLeftOrigin)

	# Detect the "oil palm" loose fruit
	def cbLooseFruit(self):
		if self.image_received:
			gray= cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
			hsv = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV)

			# construct a mask for the color "green", then perform
			# a series of dilations and erosions to remove any small
			# blobs left in the mask
			mask1 = cv2.inRange(hsv, self.lower_red, self.upper_red)
			mask2 = cv2.inRange(hsv, self.lowerRed, self.upperRed)
			mask = mask1 + mask2
			mask = cv2.erode(mask, None, iterations=1)
			mask = cv2.dilate(mask, None, iterations=1)
			
			# compute the exact Euclidean distance from every binary
			# pixel to the nearest zero pixel, then find peaks in this
			# distance map
			D = ndimage.distance_transform_edt(mask)
			localMax = peak_local_max(D, indices=False, min_distance=50,
				labels=mask)
				
			# perform a connected component analysis on the local peaks,
			# using 8-connectivity, then appy the Watershed algorithm
			markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
			self.labels = watershed(-D, markers, mask=mask)
			
			# loop over the unique labels returned by the Watershed
			# algorithm
			for label in np.unique(self.labels):
				# if the label is zero, we are examining the 'background'
				# so simply ignore it
				if label == 0:
					continue
					
				# otherwise, allocate memory for the label region and draw
				# it on the mask
				mask = np.zeros(gray.shape, dtype="uint8")
				mask[self.labels == label] = 255
				
				# detect contours in the mask and grab the largest one
				cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
					cv2.CHAIN_APPROX_SIMPLE)
				cnts = imutils.grab_contours(cnts)
				c = max(cnts, key=cv2.contourArea)
				
				# draw a circle enclosing the object
				((x, y), r) = cv2.minEnclosingCircle(c)
				cv2.circle(self.cv_image, (int(x), int(y)), int(r), (0, 255, 0), 2)
#				cv2.putText(self.cv_image, "#{}".format(label), (int(x) - 10, int(y)),
#					cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

			self.cbInfo()
			self.cbShowImage()

			# Allow up to one second to connection
			rospy.sleep(0.1)
		else:
			rospy.logerr("No images recieved")

	# rospy shutdown callback
	def cbShutdown(self):
		try:
			rospy.logwarn("[Robot1] Color Detection node [OFFLINE]")
		finally:
			cv2.destroyAllWindows()

if __name__ == '__main__':

	# Initializing your ROS Node
	rospy.init_node('robot1_color_based_detection', anonymous=False)
	color = ColorDetection()

	# Camera preview
	while not rospy.is_shutdown():
		color.cbLooseFruit()
