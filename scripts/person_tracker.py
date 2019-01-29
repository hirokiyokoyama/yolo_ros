#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from geometry_msgs.msg import PointStamped
from yolo_ros.msg import Object3dArray
from filterpy.kalman import FadingKalmanFilter
from yolo_ros.libs import TreeReader
import tf2_ros
import tf2_geometry_msgs

from visualization_msgs.msg import Marker, MarkerArray
def create_marker(x, P):
    ma = MarkerArray()
    ma.markers.append(Marker())
    ma.markers.append(Marker())

    val, vec = np.linalg.eig(P[:2,:2])
    th = np.arctan2(vec[1,0], vec[0,0])

    ma.markers[0].header.stamp = rospy.Time.now()
    ma.markers[0].header.frame_id = 'base_footprint'
    ma.markers[0].type = Marker.CYLINDER
    ma.markers[0].action = Marker.ADD
    ma.markers[0].ns = 'position'

    ma.markers[0].pose.position.x = x[0]
    ma.markers[0].pose.position.y = x[1]
    ma.markers[0].pose.position.z = 0
    ma.markers[0].pose.orientation.x = 0.
    ma.markers[0].pose.orientation.y = 0.
    ma.markers[0].pose.orientation.z = np.sin(th/2)
    ma.markers[0].pose.orientation.w = np.cos(th/2)
    ma.markers[0].scale.x = val[0]
    ma.markers[0].scale.y = val[1]
    ma.markers[0].scale.z = 0.02

    ma.markers[0].color.r = 1.0
    ma.markers[0].color.g = 0.5
    ma.markers[0].color.b = 0.0
    ma.markers[0].color.a = 0.3

    ma.markers[1].header.stamp = rospy.Time.now()
    ma.markers[1].header.frame_id = 'base_footprint'
    ma.markers[1].type = Marker.ARROW
    ma.markers[1].action = Marker.ADD
    ma.markers[1].ns = 'velocity'

    ma.markers[1].points.append(Point())
    ma.markers[1].points.append(Point())
    ma.markers[1].points[0].x = x[0]
    ma.markers[1].points[0].y = x[1]
    ma.markers[1].points[0].z = 0.
    ma.markers[1].points[1].x = x[0]+x[2]
    ma.markers[1].points[1].y = x[1]+x[3]
    ma.markers[1].points[1].z = 0.
    ma.markers[1].scale.x = .05
    ma.markers[1].scale.y = .2

    ma.markers[1].color.r = 1.0
    ma.markers[1].color.g = 0.2
    ma.markers[1].color.b = 0.0
    ma.markers[1].color.a = 0.7

    return ma

kf = FadingKalmanFilter(1.01, 4, 2)
# Measurement matrix 観測行列
kf.H = np.array([[1,0,0,0],
                 [0,1,0,0]],
                dtype=np.float)

tfBuffer = tf2_ros.Buffer()
tfListener = tf2_ros.TransformListener(tfBuffer)

after = 0
x_param = 0.6
y_param = 0.6
xms_param = 0.4
yms_param = 0.4

def callback(objects):
    global after, x, y, xms, yms
    print "callback"
    now = int(str(rospy.Time.now().secs)+str(rospy.Time.now().nsecs))
    dt = now - after
    print dt , after , now

    # Transition matrix
    kf.F = np.array([[1,0,dt,0],
                     [0,1,0,dt],
                     [0,0,1,0],
                     [0,0,0,1]],
                    dtype=np.float)

    # Transition uncertainty 不確かさ
    kf.Q = np.array([[dt*x_param,0,0,0],
                     [0,dt*y_param,0,0],
                     [0,0,dt*xms_param,0],
                     [0,0,0,dt*yms_param]],
                    dtype=np.float)

    x, P = kf.get_prediction()
    kf.predict()

    for obj in objects.objects:
        personness = tr.probability(obj.class_probability, 'person')
        personness *= obj.objectness
        prob = personness
        #TODO: reduce prob according to (row, col)
        if prob < 0.3:
            continue
        # Measurement uncertainty
        kf.R = np.array([[1/prob,0],
                         [0,1/prob]],
                        dtype=np.float)
        #tf from objects.header.frame_id to base_footprint
        ps = PointStamped()
        ps.header.stamp = objects.header.stamp
        ps.header.frame_id = objects.header.frame_id
        ps.point = obj.center
        pos = tfBuffer.transform(ps, 'base_footprint').point
        kf.update(np.array([[pos.x], [pos.y]]))

        out_true = PointStamped()
        out_true.header.frame_id = 'base_footprint'
        out_true.header.stamp = rospy.Time.now()
        out_true.point.x = pos.x
        out_true.point.y = pos.y
        #out_true.point.z = pos.z
        pub_true.publish(out_true)
        
        after =  int(str(rospy.Time.now().secs)+str(rospy.Time.now().nsecs))

    out = PointStamped()
    out.header.frame_id = 'base_footprint'
    out.header.stamp = rospy.Time.now()
    out.point.x = x[0,0]
    out.point.y = x[1,0]
    pub.publish(out)
    pub_marker.publish(create_marker(x,P))

rospy.init_node('person_tracker')
print 'before callback'
tr = TreeReader()
sub = rospy.Subscriber('objects3d', Object3dArray, callback)
pub = rospy.Publisher('person', PointStamped, queue_size=10)
pub_true = rospy.Publisher('true_person', PointStamped, queue_size=10)
pub_marker = rospy.Publisher('person_marker', MarkerArray, queue_size=10)
rospy.spin()
