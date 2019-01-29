#!/usr/bin/env python

import cv2
import rospy
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from message_filters import Subscriber

bridge = CvBridge()

def callback(image):
    try:
        cv_image = bridge.imgmsg_to_cv2(image)
    except CvBridgeError as e:
        rospy.logerr(e)


    cv_image = cv2.resize(cv_image, (480,480))
    cv2.imshow('depth_image', cv_image)
    cv2.waitKey(1)

if __name__=='__main__':
    rospy.init_node('depth_visualizer')
    image_sub = rospy.Subscriber('image', Image, callback)
    rospy.spin()
