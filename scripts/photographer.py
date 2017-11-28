#!usr/bin/env python

import sys
import rospy
import cv2
# import usb_cam
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from os.path import expancuser, join

# photographer.py

# Code Modified from :
# https://github.com/markwsilliman/turtlebot/blob/master/take_photo.py
# http://wiki.ros.org/turtlebot/Tutorials/indigo/Create%20your%20First%20Rapp

class TurtlePhoto:
	def __init__(self):
		self.bridge = CvBridge()
		self.image_received = False
		
		img_topic = "/camera/rgb/image_raw"
		self.image_sub = rospy.Subscriber(img_topic, Image, self.callback)

	def callback(self, data):
		# convert to OpenCV format
		try:
			cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
		except CvBridgeError as err:
			print(err)
		
		self.image_received = True
		self.image = cv_image

	def take_picture(self, img_title):
		if self.image_received:
			cv2.imwrite(img_title, self.image)
			return True
		else:
			return False

def photographer():
	rospy.init_node('photographer', anonymous=True)
	
	pic_num = 1

	turt_path = rospy.get_param('~turtle_path', './turtle_pictures/')
	usb_path = rospy.get_param('~usb_path', './usb_pictures/')
	freq = rospy.get_param('~frequency', 10)
	topic_name = rospy.get_param('~topic_name', 'photographer')
	img_title = rospy.get_param('~image_title', 'test_photo.jpg')
	pub = rospy.Publisher(topic_name, Image, queue_size=10)

	img_title = "%04d" % (pic_num,)

	turt = TurtlePhoto()
#	usb = USBPhoto()

	r = rospy.Rate(freq) # 10 Hz

	while not rospy.is_shutdown():
		if turt.take_picture(img_title):
			rospy.loginfo('Saved image' + img_title + ' to ' + turt_path)
			pic_num += 1
#		elif usb.take_picture(img_title):
#			rospy.loginfo('Saved image' + img_title + ' to ' + usb_path)
		else:
			rospy.loginfo('No images recieved')

		r.sleep()


if __name__ == '__main__':
	try:
		photographer()
	except rospy.ROSInterruptException: pass
