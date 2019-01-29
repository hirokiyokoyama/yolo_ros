#!/usr/bin/env python

import cv2
import rospy
import numpy as np
from yolo_ros.msg import ObjectArray, FeatureArray
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from message_filters import ApproximateTimeSynchronizer, TimeSynchronizer, Subscriber

ROW = 15
COL = 20

bridge = CvBridge()

def drawGraph(image, data, pos, color, thickness=2, height=20):
    for i,v in enumerate(data):
        left = pos[0] + thickness*i
        right = pos[0] + thickness*(i+1) - 1
        top = int(pos[1] - v*height)
        bottom = pos[1]
        cv2.rectangle(image, (left,top), (right,bottom), color, -1)

def callback(image, objects, feature):
    print "callback"
    try:
        cv_image = bridge.imgmsg_to_cv2(image)
    except CvBridgeError as e:
        rospy.logerr(e)

    feature_array = np.zeros((ROW, COL), dtype=object)
    for f in feature.features:
        feature_array[f.row, f.column] = f.data

    for obj in objects.objects:
        #rospy.loginfo('objectness={}'.format(obj.objectness))
        if obj.objectness < 0.5:
            continue
        cls = np.argmax(obj.class_probability)
        cv2.rectangle(cv_image, (int(obj.left),int(obj.top)), (int(obj.right),int(obj.bottom)), (0,255,0), 3)
        cv2.putText(cv_image, names[cls], (int(obj.left),int(obj.top)), cv2.FONT_HERSHEY_PLAIN, 1., (0,0,255))
        cv2.putText(cv_image, 'indices=({},{},{})'.format(obj.row, obj.column, obj.box_index),
                    (int(obj.left),int(obj.top+20)), cv2.FONT_HERSHEY_PLAIN, 1., (0,0,255))
        cv2.putText(cv_image, 'confidence={}'.format(obj.objectness*obj.class_probability[cls]),
                    (int(obj.left),int(obj.top+40)), cv2.FONT_HERSHEY_PLAIN, 1., (0,0,255))
        drawGraph(cv_image, obj.class_probability, (int(obj.left),int(obj.bottom)), (255,0,0))
        feat = feature_array[obj.row, obj.column]
        feat = (feat - feat.min())/(feat.max()-feat.min()) * 255
        feat = np.repeat(feat.astype(np.int32),3).reshape([32,-1,3])
        cv_image[int(obj.top):int(obj.top+32),int(obj.left):int(obj.left+feat.shape[1]),:] = feat
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
feat_sub = Subscriber('feature', FeatureArray)
sub = TimeSynchronizer([object_sub, feat_sub], 10)
sub = ApproximateTimeSynchronizer([image_sub, sub], 10, .5)
sub.registerCallback(callback)
rospy.spin()
