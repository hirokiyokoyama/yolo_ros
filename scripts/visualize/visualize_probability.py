#!/usr/bin/env python
# -*- coding: utf-8 -*-

#memo
'apple = [42, 6179]'


import cv2
import rospy
import numpy as np
from yolo_tf.msg import ObjectArray, FeatureArray
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from message_filters import ApproximateTimeSynchronizer, TimeSynchronizer, Subscriber
from yolo_tf.libs import TreeReader
from collections import Counter

bridge = CvBridge()
print 'test1'
tr = TreeReader(srv_name='get_names')
print 'test2'

#調べるオブジェクト名

n_obj = 'bell pepper'
test_obj = [i for i,name in enumerate(tr.names) if name==n_obj]
print test_obj
id_obj = test_obj[0]

'''
調べたオブジェクト
'pepper':[160],
'bell pepper'[166]

'''

def callback(image, objects):
    try:
        cv_image = bridge.imgmsg_to_cv2(image)
    except CvBridgeError as e:
        rospy.logerr(e)

    for obj in objects.objects:
        class_probability = list(obj.class_probability[:])
        #tr.remove(class_probability,81)
        #rospy.loginfo('objectness={}'.format(obj.objectness))
        left = int(obj.left)
        top = int(obj.top)
        right = int(obj.right)
        bottom = int(obj.bottom)
        # オブジェクかどうかの判別
        # if obj.objectness < 0.1:
        #    continue
        name = None
        prob = 0.
        number = 0
        count = 0
        # オブジェクであるかの値を掛け合わせる
        s_obj = tr.probability(class_probability,id_obj) * obj.objectness
        print n_obj,':',s_obj,'\n'

        cv2.rectangle(cv_image, (int(obj.left),int(obj.top)), (int(obj.right),int(obj.bottom)), (0,255,0), 3)
        cv2.putText(cv_image, '{}'.format(s_obj),(int(obj.left),int(obj.top+20)),cv2.FONT_HERSHEY_PLAIN, 1., (0,0,255))
        cv2.putText(cv_image, n_obj, (int(obj.left),int(obj.top)), cv2.FONT_HERSHEY_PLAIN, 1., (0,0,255))

    cv2.imshow(image_sub.resolved_name, cv_image)
    cv2.waitKey(1)
        


rospy.init_node('yolo9000_visualizer_id')
#後でリマッピング
image_sub = Subscriber('/hsrb/head_rgbd_sensor/rgb/image_color', Image)
object_sub = Subscriber('objects', ObjectArray)
sub = ApproximateTimeSynchronizer([image_sub, object_sub], 10, 10)
sub.registerCallback(callback)
print 'TreeReader created'


rospy.spin()
