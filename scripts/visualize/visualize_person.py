#!/usr/bin/env python

import cv2
import rospy
import numpy as np
from yolo_ros.msg import ObjectArray, FeatureArray
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from message_filters import ApproximateTimeSynchronizer, TimeSynchronizer, Subscriber
from yolo_ros.libs import TreeReader
import time

bridge = CvBridge()
tr = TreeReader()

def callback(image, objects):
    print 'callback'
    try:
        cv_image = bridge.imgmsg_to_cv2(image, 'bgr8')
    except CvBridgeError as e:
        rospy.logerr(e)

    for obj in objects.objects:
        #rospy.loginfo('objectness={}'.format(obj.objectness))
        #if obj.objectness < 0.3:
         #   continue
	prob = obj.objectness * tr.probability(obj.class_probability, "person")
        if prob < 0.3:
	    continue
        print prob
        
        cv2.rectangle(cv_image,(int(obj.left),int(obj.top)), (int(obj.right),int(obj.bottom)), (0,255,0), 3)
        
    cv2.imshow(image_sub.resolved_name, cv_image)
    cv2.waitKey(1)
    
rospy.init_node('yolo9000_visualizer')
image_sub = Subscriber('image', Image)
object_sub = Subscriber('objects', ObjectArray)
sub = ApproximateTimeSynchronizer([image_sub, object_sub], 100, 100)
sub.registerCallback(callback)
print 'TreeReader created'
rospy.spin()
