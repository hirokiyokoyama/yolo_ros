#!/usr/bin/env python

import cv2
import rospy
import numpy as np
from yolo_tf.msg import ObjectArray, FeatureArray
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from message_filters import TimeSynchronizer, Subscriber
from yolo_tf.srv import GetNames

bridge = CvBridge()

def callback(objects, image):
    try:
        cv_image = bridge.imgmsg_to_cv2(image)
    except CvBridgeError as e:
        rospy.logerr(e)

    #print 'Message arrived.'
    for obj in objects.objects:
        if obj.objectness < 0.3:
            continue
        personness = obj.class_probability[person_ind]
        ind = person_ind
        while nodes[ind].parent > 0:
            ind = nodes[ind].parent
            personness *= obj.class_probability[ind]
        #rospy.loginfo('P(person|object)={}'.format(personness))
        if personness < 0.5:
            continue
        subcls_prob = np.array(obj.class_probability)[np.array(children_ind)]
        sorted_inds = np.argsort(subcls_prob)[::-1]
        cv2.rectangle(cv_image, (int(obj.left),int(obj.top)), (int(obj.right),int(obj.bottom)), (0,255,0), 3)
        text_y = 0
        for ind in sorted_inds[:10]:
            desc = '{}: {}'.format(names[children_ind[ind]], subcls_prob[ind])
            cv2.putText(cv_image, desc,
                        (int(obj.left),int(obj.top+text_y)), cv2.FONT_HERSHEY_PLAIN, 1., (0,0,255))
            text_y += 20
            rospy.loginfo(desc)

        cv2.putText(cv_image, 'objectness={}'.format(obj.objectness),
                    (int(obj.left),int(obj.top+text_y)), cv2.FONT_HERSHEY_PLAIN, 1., (0,0,255))
    cv2.imshow(image_sub.resolved_name, cv_image)
    cv2.waitKey(1)
    
global names, nodes
rospy.init_node('yolo_visualizer')
rospy.wait_for_service('get_names')
try:
    get_names = rospy.ServiceProxy('get_names', GetNames)
    res = get_names()
    names = res.names
    nodes = res.tree_nodes
except rospy.ServiceException as e:
    rospy.logerr('Service call failed: {}'.format(e))
person_ind = 5177
children_ind = nodes[person_ind].children

image_sub = Subscriber('image', Image)
object_sub = Subscriber('objects', ObjectArray)
sub = TimeSynchronizer([object_sub, image_sub], 10)
sub.registerCallback(callback)
rospy.spin()
