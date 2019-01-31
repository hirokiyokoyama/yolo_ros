#!/usr/bin/env python

import cv2
import rospy
import numpy as np
from yolo_ros.msg import ObjectArray
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from message_filters import ApproximateTimeSynchronizer, TimeSynchronizer, Subscriber
from yolo_ros.libs import TreeReader
import time

bridge = CvBridge()
tr = TreeReader()
image_to_show = None

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

def obj_iou(obj1, obj2):
    return box_iou((obj1.left + obj1.right)/2,
                   (obj1.top + obj1.bottom)/2,
                   obj1.right - obj1.left,
                   obj1.bottom - obj1.top,
                   (obj2.left + obj2.right)/2,
                   (obj2.top + obj2.bottom)/2,
                   obj2.right - obj2.left,
                   obj2.bottom - obj2.top)

def callback(image, objects):
    try:
        cv_image = bridge.imgmsg_to_cv2(image, 'bgr8')
    except CvBridgeError as e:
        rospy.logerr(e)

    objs = []
    for obj in objects.objects:
        if obj.objectness < 0.2:
            continue
        name = None
        prob = 1.
        for _, _name, _prob in tr.trace_max(obj.class_probability):
            if obj.objectness * _prob > 0.2:
                name = _name
                prob = _prob
            else:
                name = _name
                break

        if objs:
            iou = [obj_iou(obj, obj2) for obj2, _, _ in objs]
            i = np.argmax(iou)
            if iou[i] > .5:
                if objs[i][0].objectness < obj.objectness:
                    objs[i] = (obj, name, prob)
            else:
                objs.append((obj, name, prob))
        else:
            objs.append((obj, name, prob))

    for obj, name, prob in objs:
        cv2.rectangle(cv_image, (int(obj.left),int(obj.top)), (int(obj.right),int(obj.bottom)), (64,128,256), 2)
        text = '{}: {}%'.format(name, int(obj.objectness*prob*100))
        font = cv2.FONT_HERSHEY_PLAIN
        font_scale = 1.
        thickness = 1
        (w, h), base_line = cv2.getTextSize(text, font, font_scale, thickness)
        h += base_line
        cv2.rectangle(cv_image, (int(obj.left),int(obj.top)), (int(obj.left+w),int(obj.top+h)), (64,128,256), -1)
        cv2.putText(cv_image, text,
                    (int(obj.left),int(obj.top+h)), font, font_scale, (0,0,0), thickness)

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
