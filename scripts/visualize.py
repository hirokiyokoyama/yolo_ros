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
image_to_show = None

def callback(image, objects):
    print 'callback'
    try:
        cv_image = bridge.imgmsg_to_cv2(image, 'bgr8')
    except CvBridgeError as e:
        rospy.logerr(e)

    for obj in objects.objects:
        rospy.loginfo('objectness={}'.format(obj.objectness))
        if obj.objectness < 0.35:
            continue
        name = None
        prob = 0.
        for _, _name, _prob in tr.trace_max(obj.class_probability):
            if _prob > 0.1:
                name = _name
                prob = _prob
            else:
                name = _name
                break
        
        cv2.rectangle(cv_image, (int(obj.left),int(obj.top)), (int(obj.right),int(obj.bottom)), (0,255,0), 3)
        cv2.putText(cv_image, '{}: {}%'.format(name, int(obj.objectness*prob*100)),
                    (int(obj.left),int(obj.top)), cv2.FONT_HERSHEY_PLAIN, 1., (0,0,255))
        print '{} {}%'.format(name, int(obj.objectness*prob*100))

    global image_to_show
    image_to_show = cv_image
    
rospy.init_node('yolo9000_visualizer')
image_sub = Subscriber('image', Image)
object_sub = Subscriber('objects', ObjectArray)
sub = ApproximateTimeSynchronizer([image_sub, object_sub], 100, 100)
sub.registerCallback(callback)

while not rospy.is_shutdown():
    img = image_to_show
    if img is not None:
        cv2.imshow('YOLO detection', img)
        cv2.waitKey(1)
