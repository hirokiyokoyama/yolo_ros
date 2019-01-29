#!/usr/bin/env python

import cv2
import rospy
import numpy as np
from yolo_ros.msg import ObjectArray, Object3dArray, Object3dDesc
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
from message_filters import ApproximateTimeSynchronizer, Subscriber
from image_geometry import PinholeCameraModel

bridge = CvBridge()

NUM_BINS = 20

def callback(image, objects, camera):
    print "callback"
    try:
        cv_image = bridge.imgmsg_to_cv2(image)
    except CvBridgeError as e:
        rospy.logerr(e)
    valid_depths = cv_image.flat[np.isfinite(cv_image.flat)]
    depth_hist, depth_bins = np.histogram(valid_depths, NUM_BINS)

    pinhole = PinholeCameraModel()
    pinhole.fromCameraInfo(camera)

    out = Object3dArray()
    out.header.stamp = objects.header.stamp
    out.header.frame_id = objects.header.frame_id
    bin_votes = np.zeros(NUM_BINS, dtype=np.int32)
    for obj in objects.objects:
        #if obj.objectness < 0.5:
        #    continue
        top = min(max(0, int(obj.top)), cv_image.shape[0])
        bottom = min(max(0, int(obj.bottom)), cv_image.shape[0])
        left = min(max(0, int(obj.left)), cv_image.shape[1])
        right = min(max(0, int(obj.right)), cv_image.shape[1])
        cx = int((obj.bottom+obj.top)/2)
        cy = int((obj.right+obj.left)/2)
        width = int(obj.bottom-obj.top)
        height = int(obj.right-obj.left)
        #depth_cropped = cv_image[top:bottom, left:right]
        depth_cropped = cv_image
        bin_votes[:] = 0

        #TODO: make a map from bin to nearest peak and use it for all pixels
        for x in range(max(0, cx-width/4), min(cx+width/4, cv_image.shape[1])):
            for y in range(max(0, cy-height/4), min(cy+height/4, cv_image.shape[0])):
                if not np.isfinite(depth_cropped[y,x]):
                    continue
                bin_ind = np.where(depth_cropped[y,x] <= depth_bins)[0][0]-1
                ##print '({},{}): {}'.format(x,y,bin_ind)
                # find the nearest peak
                while True:
                    neighbors = depth_hist[max(0,bin_ind-1):(bin_ind+2)]
                    if len(neighbors) == 0 or \
                       neighbors.max() == depth_hist[bin_ind]:
                        bin_ind = max(0, min(NUM_BINS-1, bin_ind))
                        break
                    if neighbors[0] < neighbors[-1]:
                        bin_ind += 1
                    else:
                        bin_ind -= 1
                # TODO: deal with consecutive peaks
                bin_votes[bin_ind] += 1
        if bin_votes.max() == 0:
            continue
        bin_ind = bin_votes.argmax()
        depth = (depth_bins[bin_ind]+depth_bins[bin_ind+1])/2

        obj3d = Object3dDesc()
        center = np.array(pinhole.projectPixelTo3dRay((cx,cy))) * depth
        left = np.array(pinhole.projectPixelTo3dRay((left,cy))) * depth
        right = np.array(pinhole.projectPixelTo3dRay((right,cy))) * depth
        top = np.array(pinhole.projectPixelTo3dRay((cx,top))) * depth
        bottom = np.array(pinhole.projectPixelTo3dRay((cx,bottom))) * depth
        obj3d.center.x, obj3d.center.y, obj3d.center.z = center
        obj3d.width = np.sqrt(((right-left)**2).sum())
        obj3d.width = np.sqrt(((bottom-top)**2).sum())
        obj3d.objectness = obj.objectness
        obj3d.class_probability = obj.class_probability

        out.objects.append(obj3d)
    pub.publish(out)
    print 'publish'

rospy.init_node('depth_merger')
image_sub = Subscriber('image', Image)
object_sub = Subscriber('objects', ObjectArray)
camera_sub = Subscriber('camera', CameraInfo)
sub = ApproximateTimeSynchronizer([image_sub, object_sub, camera_sub], 10, .5)
sub.registerCallback(callback)
pub = rospy.Publisher('objects3d', Object3dArray, queue_size=10)
rospy.spin()
