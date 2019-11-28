#!/usr/bin/env python

import tensorflow as tf
from nets import Yolo, YoloOutputCodec
from box import obj_iou

import cv2
import numpy as np

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from yolo_ros.msg import ObjectArray, ObjectDesc
from yolo_ros.srv import DetectObjects, DetectObjectsResponse
from yolo_ros.cfg import YoloDetectorConfig

from dynamic_reconfigure.server import Server

class YoloDetector:
    def __init__(self, ckpt, image_size=None, visualization=False):
        self.image_size = image_size
        self.model = Yolo(num_classes=1, num_boxes=1)
        self.model.load_weights(ckpt_file)
        rospy.loginfo('Network was restored from {}.'.format(ckpt_file))

        self.codec = None
        self.bridge = CvBridge()
        self.threshold = 0.3
        Server(YoloDetectorConfig, self.config)

        self.visualization = visualization
        self.visualized = None

        self.obj_pub = rospy.Publisher('objects', ObjectArray, queue_size=1)
        rospy.Service('detect_objects', DetectObjects, self.detect_objects)
        self.image_sub = rospy.Subscriber('image', Image, self.callback,
                                          queue_size=1, buff_size=1024*1024*4)

    def config(self, config, level):
        self.threshold = config["threshold"]
        return config

    def callback(self, imgmsg):
        if self.obj_pub.get_num_connections() > 0 or self.visualization:
            res, cvimg = self.process_imgmsg(imgmsg)
            self.obj_pub.publish(res)

            if self.visualization:
                self.visualized = self.visualize(cvimg, res)

    def detect_objects(self, req):
        res, _ = self.process_imgmsg(req.image)
        return DetectObjectsResponse(objects=res)

    @tf.function
    def _run_tf(self, img):
        image_batch = tf.expand_dims(tf.cast(img, tf.float32), 0) / 255. * 2. -1.
        if self.image_size is not None:
            image_batch = tf.image.resize(image_batch, self.image_size[::-1])
        return self.model(image_batch)

    def process_imgmsg(self, imgmsg):
        if self.codec is None:
            if self.image_size is None:
                image_size = [imgmsg.width, imgmsg.height]
            else:
                image_size = self.image_size
            self.codec = YoloOutputCodec(
                image_size = image_size,
                anchors = [[50.,50.]]
            )

        try:
            cv_image = self.bridge.imgmsg_to_cv2(imgmsg, 'rgb8')
        except CvBridgeError as e:
            rospy.logerr(e)

        t0 = rospy.Time.now()
        pred = self._run_tf(cv_image)
        t1 = rospy.Time.now()
        rospy.loginfo('Processing time: {} ms'.format((t1-t0).to_sec()*1000))
        msg = self.process_prediction(pred)
        msg.header.stamp = imgmsg.header.stamp
        msg.header.frame_id = imgmsg.header.frame_id
        return msg, cv_image

    def process_prediction(self, prediction):
        msg = ObjectArray()
        x, y, w, h, obj, cls = map(lambda x: x[0].numpy(), self.codec.decode(prediction))

        rospy.loginfo('Maximum objectness: {}/{}'.format(obj.max(), self.threshold))

        box_index = 0
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                for k in range(x.shape[2]):
                    if obj[i,j,k] < self.threshold:
                        continue
                    objmsg = ObjectDesc()
                    objmsg.row = i
                    objmsg.column = j
                    objmsg.box_index = box_index
                    box_index += 1
                    objmsg.top = y[i,j,k]
                    objmsg.left = x[i,j,k]
                    objmsg.bottom = y[i,j,k]+h[i,j,k]
                    objmsg.right = x[i,j,k]+w[i,j,k]
                    objmsg.objectness = obj[i,j,k]
                    objmsg.class_probability = cls[i,j,k]
                    msg.objects.append(objmsg)
        return msg

    def visualize(self, img, msg):
        img = img.copy()
        objs = []
        for obj in msg.objects:
            if objs:
                iou = [obj_iou(obj, obj2) for obj2 in objs]
                i = np.argmax(iou)
                if iou[i] > .5:
                    if objs[i].objectness < obj.objectness:
                        objs[i] = obj
                else:
                    objs.append(obj)
            else:
                objs.append(obj)

        for obj in objs:
            cv2.rectangle(img, (int(obj.left),int(obj.top)), (int(obj.right),int(obj.bottom)), (64,128,256), 2)

        return img        

if __name__ == '__main__':
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for k in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[k], True)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, help='ckpt file.')
    args, _ = parser.parse_known_args()

    rospy.init_node('yolo')
    yd = YoloDetector(args.ckpt, visualization=True)

    #rospy.spin()
    while not rospy.is_shutdown():
        img = yd.visualized
        if img is not None:
            cv2.imshow('YOLO detection', img[:,:,::-1])
        cv2.waitKey(1)

