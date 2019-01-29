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

def box_overlap(x1, w1, x2, w2):
    l1 = x1 - w1/2
    l2 = x2 - w2/2
    left = max(l1, l2)
    r1 = x1 + w1/2
    r2 = x2 + w2/2
    right = min(r1, r2)
    return right - left

def box_intersection(ax, ay, aw, ah, bx, by, bw, bh):
    w = box_overlap(ax, aw, bx, bw)
    h = box_overlap(ay, ah, by, bh)
    if w < 0 or h < 0:
        return 0
    return w*h

def box_union(ax, ay, aw, ah, bx, by, bw, bh):
    i = box_intersection(ax, ay, aw, ah, bx, by, bw, bh)
    u = aw*ah + bw*bh - i
    return u

def box_iou(ax, ay, aw, ah, bx, by, bw, bh):
    i = box_intersection(ax, ay, aw, ah, bx, by, bw, bh)
    u = aw*ah + bw*bh - i
    return float(i)/u

def callback(image, objects):
    obj_coordinate = []
    try:
        cv_image = bridge.imgmsg_to_cv2(image, 'bgr8')
    except CvBridgeError as e:
        rospy.logerr(e)

    for obj in objects.objects:
        #rospy.loginfo('objectness={}'.format(obj.objectness))
        if obj.objectness < 0.35:
            continue
        name = None
        prob = 0.
        for _, _name, _prob in tr.trace_max(obj.class_probability):
            if _prob > 0.1:
                name = _name
                prob = _prob
                obj_coordinate.append([[int(obj.left),int(obj.top)] ,[int(obj.right),int(obj.bottom)]])
            else:
                name = _name
                break
        
        #cv2.rectangle(cv_image, (int(obj.left),int(obj.top)), (int(obj.right),int(obj.bottom)), (0,255,0), 1)
        
        
        cv2.putText(cv_image, 'name={}'.format(name), (int(obj.left),int(obj.top)), cv2.FONT_HERSHEY_PLAIN, 1., (0,0,255))
        cv2.putText(cv_image, 'indices=({},{},{})'.format(obj.row, obj.column, obj.box_index),
                    (int(obj.left),int(obj.top+20)), cv2.FONT_HERSHEY_PLAIN, 1., (0,0,255))
        cv2.putText(cv_image, 'confidence={}'.format(obj.objectness*prob),
                    (int(obj.left),int(obj.top+40)), cv2.FONT_HERSHEY_PLAIN, 1., (0,0,255))
        
        if name != None:
            print(name , float(obj.objectness*prob))
        
    result = []
    for x in obj_coordinate:
        if x not in result:
            result.append(x)
    
    for i in range(len(result)):
        if len(result) == 1:
            print 'continue'
            continue
        try:
            iou = box_iou(result[i][0][0], result[i][0][1], result[i][1][0], result[i][1][1], result[i+1][0][0], result[i+1][0][1], result[i+1][1][0], result[i+1][1][1])
            print iou
        except:
            print 'dame'
        if iou >= 0.3:
            cv2.rectangle(cv_image, (int(result[i][0][0]), int(result[i][0][1])), (int(result[i][1][0]), int(result[i][1][1])), (0,0,255), 1)
        elif iou == 0:
            cv2.rectangle(cv_image, (int(result[i][0][0]), int(result[i][0][1])), (int(result[i][1][0]), int(result[i][1][1])), (0,255,0), 1)
    cv2.imshow(image_sub.resolved_name, cv_image)
    cv2.waitKey(1)
    
rospy.init_node('yolo9000_visualizer')
image_sub = Subscriber('image', Image)
object_sub = Subscriber('objects', ObjectArray)
sub = ApproximateTimeSynchronizer([image_sub, object_sub], 100, 100)
sub.registerCallback(callback)
print 'TreeReader created'
rospy.spin()
