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
from std_msgs.msg import String, Int32
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo

from cv_bridge import CvBridge
from cv_bridge import CvBridgeError

from common_tracking_application.objcenter import objCenter
from common_face_application.msg import objCenter as objCoord

class ColorDetectionROI:

	def __init__(self):

		rospy.logwarn("[Robot1] Color Detection (ROI) node [ONLINE]")

		self.bridge = CvBridge()
		self.objectCoord = objCoord()

		# define the lower and upper boundaries of the "oil palm"
		# in the HSV color space, then initialize the
		# list of tracked points
		self.lower_red = (0, 120, 70)
        	self.upper_red = (10, 255, 255)
        	self.lowerRed = (170, 120, 70)
        	self.upperRed = (180, 255, 255)

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
			
		# Subscribe to Int32 msg
		possible_fruit_topic = "/possible_fruit_robot1"
		self.possible_fruit_sub = rospy.Subscriber(possible_fruit_topic, Int32, self.cbPossibleFruit)
			
		# Publish to objCenter msg
		objCoord_topic = "/objCoord_robot1"
		self.objCoord_pub = rospy.Publisher(objCoord_topic, objCoord, queue_size=10)

		# Allow up to one second to connection
		rospy.sleep(1)

	# Convert image to OpenCV format
	def cbImage(self, msg):

		try:
			self.cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
			self.cv_image = cv2.flip(self.cv_image, 1)
			self.cv_image_copy = self.cv_image.copy()
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
		
	def cbPossibleFruit(self, msg):
	
		self.possible_fruit = msg.data

	# Show the output frame
	def cbShowImage(self):

		cv2.imshow("[Robot1] Loose Fruit Detector (ROI)", self.cv_image)
		cv2.waitKey(1)

	# Image information callback
	def cbInfo(self):

		fontFace = cv2.FONT_HERSHEY_DUPLEX
		fontScale = 0.5
		color = (255, 255, 255)
		thickness = 1
		lineType = cv2.LINE_AA
		bottomLeftOrigin = False # if True (text upside down)

#		self.timestr = time.strftime("%Y%m%d-%H:%M:%S")

#		cv2.putText(self.cv_image, "{}".format(self.timestr), (10, 20), 
#			fontFace, fontScale, color, thickness, lineType, 
#			bottomLeftOrigin)
		cv2.putText(self.cv_image, "Possible Fruit: %d"% (self.possible_fruit), (5, 20), 
			fontFace, fontScale, (0, 0, 255), thickness, lineType, 
			bottomLeftOrigin)
		cv2.putText(self.cv_image, "(%d, %d)" % (self.objX, self.objY), 
			(self.imgWidth-100, 20), fontFace, fontScale, 
			color, thickness, lineType, bottomLeftOrigin)
#		cv2.putText(self.cv_image, "Sample", (10, self.imgHeight-10), 
#			fontFace, fontScale, color, thickness, lineType, 
#			bottomLeftOrigin)
#		cv2.putText(self.cv_image, "(%d, %d)" % (self.imgWidth, self.imgHeight), 
#			(self.imgWidth-100, self.imgHeight-10), fontFace, fontScale, 
#			color, thickness, lineType, bottomLeftOrigin)

	# Detect the "oil palm" loose fruit
	def cbLooseFruit(self):
		if self.image_received:
			gray= cv2.cvtColor(self.cv_image_copy, cv2.COLOR_BGR2GRAY)
			hsv = cv2.cvtColor(self.cv_image_copy, cv2.COLOR_BGR2HSV)

			# construct a mask for the color "green", then perform
			# a series of dilations and erosions to remove any small
			# blobs left in the mask
			mask1 = cv2.inRange(hsv, self.lower_red, self.upper_red)
			mask2 = cv2.inRange(hsv, self.lowerRed, self.upperRed)
			mask = mask1 + mask2
			mask = cv2.erode(mask, None, iterations=1)
			mask = cv2.dilate(mask, None, iterations=1)

			# find contours in the mask and initialize the current
			# (x, y) center of the ball
			cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
				cv2.CHAIN_APPROX_SIMPLE)
			cnts = imutils.grab_contours(cnts)

			# only proceed if at least one contour was found
			objectLoc = objCenter(cnts, (self.centerX, self.centerY))
			((self.objX, self.objY), rect) = objectLoc
			
			self.pubObjCoord()
			
			# extract the bounding box and draw it
			if rect is not None:
				cv2.circle(self.cv_image, (int(self.objX), int(self.objY)), 
					int(rect), (0, 255, 255), 2)

#			# only proceed if at least one contour was found
#			if len(cnts) > 0:
#				# find the largest contour in the mask, then use
#				# it to compute the minimum enclosing circle and
#				# centroid
#				c = max(cnts, key=cv2.contourArea)
#				((x, y), radius) = cv2.minEnclosingCircle(c)
##				M = cv2.moments(c)
##				center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
#				
#				# only proceed if the radius meets a minimum size
#				if radius > 10:
#					# draw the circle and centroid on the frame,
#					# then update the list of tracked points
#					cv2.circle(self.cv_image, (int(x), int(y)), int(radius),
#						(0, 255, 255), 2)
##					cv2.circle(self.cv_image, center, 5, (0, 0, 255), -1)
#					
##			# update the points queue
##			self.pts.appendleft(center)

##			# loop over the set of tracked points
##			for i in range(1, len(self.pts)):
##				# if either of the tracked points are None, ignore
##				# them
##				if self.pts[i - 1] is None or self.pts[i] is None:
##					continue
##					
##				# otherwise, compute the thickness of the line and
##				# draw the connecting lines
##				thickness = int(np.sqrt(self.buffer / float(i + 1)) * 2.5)
##				cv2.line(self.cv_image, self.pts[i - 1], self.pts[i], (0, 0, 255), thickness)

			self.cbInfo()
			self.cbShowImage()

			# Allow up to one second to connection
#			rospy.sleep(0.1)
		else:
			rospy.logerr("No images recieved")

	# Publish to objCenter msg
	def pubObjCoord(self):

		self.objectCoord.centerX = self.objX
		self.objectCoord.centerY = self.objY

		self.objCoord_pub.publish(self.objectCoord)
		
	# rospy shutdown callback
	def cbShutdown(self):
		try:
			rospy.logwarn("[Robot1] Color Detection (ROI) node [OFFLINE]")
		finally:
			cv2.destroyAllWindows()

if __name__ == '__main__':

	# Initializing your ROS Node
	rospy.init_node('color_based_detection_roi', anonymous=False)
	color = ColorDetectionROI()

	# Camera preview
	while not rospy.is_shutdown():
		color.cbLooseFruit()
