#!/usr/bin/env python

import cv2
import rospy
import numpy as np
from yolo_ros.msg import ObjectArray, FeatureArray
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from message_filters import TimeSynchronizer, Subscriber

bridge = CvBridge()

def callback(objects, image):
    try:
        cv_image = bridge.imgmsg_to_cv2(image)
    except CvBridgeError as e:
        rospy.logerr(e)

    print 'Message arrived.'
    for obj in objects.objects:
        #rospy.loginfo('objectness={}'.format(obj.objectness))
        if obj.objectness < 0.2:
            continue
        cls_inds = np.argsort(obj.class_probability)[::-1]
        print '({},{})'.format(obj.left, obj.top)
        cv2.rectangle(cv_image, (int(obj.left),int(obj.top)), (int(obj.right),int(obj.bottom)), (0,255,0), 3)
        text_y = 0
        for ind in cls_inds[:10]:
            cv2.putText(cv_image, '{}: {}'.format(names[ind], obj.class_probability[ind]),
                        (int(obj.left),int(obj.top+text_y)), cv2.FONT_HERSHEY_PLAIN, 1., (0,0,255))
            text_y += 20
        cv2.putText(cv_image, 'objectness={}'.format(obj.objectness),
                    (int(obj.left),int(obj.top+text_y)), cv2.FONT_HERSHEY_PLAIN, 1., (0,0,255))
    cv2.imshow(image_sub.resolved_name, cv_image)
    cv2.waitKey(1)

import sys
names = []
with open(sys.argv[1], 'r') as f:
    while True:
        l = f.readline()
        if not l:
            break
        names.append(l[:-1])
    
rospy.init_node('yolo_visualizer')
image_sub = Subscriber('image', Image)
object_sub = Subscriber('objects', ObjectArray)
sub = TimeSynchronizer([object_sub, image_sub], 10)
sub.registerCallback(callback)
rospy.spin()
