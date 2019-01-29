#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import rospy
import numpy as np
from yolo_ros.msg import ObjectArray, FeatureArray
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from message_filters import ApproximateTimeSynchronizer, TimeSynchronizer, Subscriber
from yolo_ros.libs import TreeReader
from collections import Counter

bridge = CvBridge()
print 'test1'
tr = TreeReader(srv_name='get_names_known')

print 'test2'

#a = ""

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
        #オブジェクではないかの判別
        if obj.objectness < 0.1:
            continue
        name = None
        prob = 0.
        number = 0
        count = 0
        for _ids, _name, _prob in tr.trace_max(class_probability):
            #その名前である確率
            number = number+15
            count = count+1
            if _prob > 0.08:
                name = _name
                prob = _prob
                ids = str(_ids)
                cv2.putText(cv_image, 'id:{}'.format(ids), (left,top+number), cv2.FONT_HERSHEY_PLAIN, 1., (255,0,0))
                cv2.putText(cv_image, '{}'.format(round(obj.objectness*prob,3)),
                           (left+60,top+number),cv2.FONT_HERSHEY_PLAIN, 1., (0,140,255))
                cv2.rectangle(cv_image, (left,top), (right,bottom), (0,255,0), 3)
                cv2.putText(cv_image, '{0}:{1}'.format(count,name), 
                           (left+110,top+number), cv2.FONT_HERSHEY_PLAIN, 1., (0,0,255))
                print 'id:{0}     prob:{1}     name{2}:{3} '.format(ids,round(obj.objectness*prob,3),count,name)
                #b += '\nid:{0}     prob:{1}     name:{2} '.format(ids,round(obj.objectness*prob,3),name)
                #print 'id:{0}     prob:{1}     name:{2} '.format(ids,round(obj.objectness*prob,3),name)


                '''
                cv2.putText(cv_image, 'indices=({},{},{})'.format(obj.row, obj.column, obj.box_index),
                            (left,top+100), cv2.FONT_HERSHEY_PLAIN, 1., (0,0,255))
                cv2.putText(cv_image, 'confidence={}'.format(obj.objectness*prob),
                            (left,top+120), cv2.FONT_HERSHEY_PLAIN, 1., (0,0,255))
                cv2.putText(cv_image, 'final name:{0}'.format(name), (left,top+80), cv2.FONT_HERSHEY_PLAIN, 1., (255,0,0))
                cv2.putText(cv_image, 'final id:{0}'.format(ids), (left,top+60), cv2.FONT_HERSHEY_PLAIN, 1., (255,0,0))
                '''


                cv2.waitKey(2)
                cv2.imshow(image_sub.resolved_name, cv_image)
                number + 100
            else:
                break
        '''
        global a
        a = b
        '''
        print 'final name:[{0}]\n'.format(name)
        #print '\n'
        # 引数の文字列をファイルに書き込む
        #print ids_list

rospy.init_node('yolo9000_visualizer_id')
#後でリマッピング
image_sub = Subscriber('/hsrb/head_rgbd_sensor/rgb/image_color', Image)
object_sub = Subscriber('known_objects', ObjectArray)
sub = ApproximateTimeSynchronizer([image_sub, object_sub], 10, 10)
sub.registerCallback(callback)
print 'TreeReader created'
'''
r = rospy.Rate(1)
while not rospy.is_shutdown():
    print a
    r.sleep()
print a
'''
rospy.spin()
