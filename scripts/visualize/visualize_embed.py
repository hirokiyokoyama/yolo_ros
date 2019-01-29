#!/usr/bin/env python

import cv2
import rospy
import numpy as np
from yolo_tf.msg import ObjectArray, FeatureArray
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from message_filters import ApproximateTimeSynchronizer, TimeSynchronizer, Subscriber
import time

bridge = CvBridge()

def callback(image, objects):
    #print 'callback'
    try:
        cv_image = bridge.imgmsg_to_cv2(image, 'bgr8')
    except CvBridgeError as e:
        rospy.logerr(e)

    filtered = filter(lambda x: x.objectness>0.3, objects.objects)

    for obj in filtered:        
        cv2.rectangle(cv_image, (int(obj.left),int(obj.top)), (int(obj.right),int(obj.bottom)), (0,255,0), 3)
        cv2.putText(cv_image, 'indices=({},{},{})'.format(obj.row, obj.column, obj.box_index),
                    (int(obj.left),int(obj.top+20)), cv2.FONT_HERSHEY_PLAIN, 1., (0,0,255))
    
    for i in xrange(len(filtered)):
        for j in xrange(i+1, len(filtered)):
            obj1 = filtered[i]
            obj2 = filtered[j]
            x1 = int((obj1.left + obj1.right)/2)
            y1 = int((obj1.top + obj1.bottom)/2)
            x2 = int((obj2.left + obj2.right)/2)
            y2 = int((obj2.top + obj2.bottom)/2)
            x = (x1 + x2)/2
            y = (y1 + y2)/2
            obj1_norm = np.linalg.norm(obj1.class_probability)
            obj2_norm = np.linalg.norm(obj2.class_probability)
            inner_product = np.dot(obj1_norm, obj2_norm)
            p = ((np.dot(obj1.class_probability, obj2.class_probability)/inner_product) + 1)/2
            cv2.line(cv_image, (x1,y1), (x2,y2), (255,0,0), 1)
            cv2.putText(cv_image, '%.2f' % p, (x, y), cv2.FONT_HERSHEY_PLAIN, 1., (0,255,0))
    cv2.imshow(image_sub.resolved_name, cv_image)
    cv2.waitKey(1)
    
rospy.init_node('yolo_embedding_visualizer')
image_sub = Subscriber('image', Image)
object_sub = Subscriber('embedded_objects', ObjectArray)
sub = ApproximateTimeSynchronizer([image_sub, object_sub], 100, 100)
sub.registerCallback(callback)
rospy.spin()
