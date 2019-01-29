#!/usr/bin/env python

import cv2
import numpy as np
import tensorflow as tf
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from nets import yolo9000
from yolo_tf.msg import ObjectArray, ObjectDesc, Feature, FeatureArray, TreeNode
from yolo_tf.srv import GetNames, GetNamesResponse
from yolo_tf.srv import DetectObjects, DetectObjectsResponse
from yolo_tf.cfg import YoloDetectorConfig

from dynamic_reconfigure.server import Server

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax_tree(x, tree):
    x = x.reshape(-1,x.shape[-1])
    for i in xrange(len(tree)-1):
        begin = tree[i]
        end = tree[i+1]
        x[:,begin:end] = np.exp(x[:,begin:end])
        x[:,begin:end] /= np.sum(x[:,begin:end])

class YoloDetector:
    def __init__(self, ckpt_file, names_file, tree_file,
                 known_ckpt_file = None, known_names_file = None,
                 embed_ckpt_file = None,
                 input_shape=(416,416),
                 num_known_boxes=3,
                 num_embed_boxes=3, embed_dim=128):
        self.ckpt_file = ckpt_file
        self.input_shape = input_shape

        self.num_classes = 9418
        self.num_boxes = 3
        self.anchor_ws = np.array([0.77871, 3.00525, 9.22725])
        self.anchor_hs = np.array([1.14074, 4.31277, 9.61974])
        self.grid_h = input_shape[0]/32
        self.grid_w = input_shape[1]/32
        self.net = yolo9000
        self.feature_tensor = 'Conv_17/Leaky:0'
        self.threshold = 0.1
        self.group_begin = [0]
        
        with open(names_file, 'r') as f:
            self.names = map(str.strip, f.readlines())
        with open(tree_file, 'r') as f:
            self.tree = [int(l.split()[1]) for l in f.readlines()]
        begin = 0
        end = 0
        while begin < len(self.tree):
            end = begin+1
            while end < len(self.tree) and self.tree[end] == self.tree[begin]:
                end += 1
            begin = end
            self.group_begin.append(begin)

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('image', Image, self.callback)
        self.obj_pub = rospy.Publisher('objects', ObjectArray, queue_size=10)
        self.feat_pub = rospy.Publisher('features', FeatureArray, queue_size=10)
        self.sess = None
        self.sess_known = None
        self.sess_embed = None

        rospy.Service('detect_objects', DetectObjects, self.detect_objects)
        rospy.Service('get_names', GetNames, self.get_names)

        self.known_ckpt_file = known_ckpt_file
        if known_ckpt_file is not None:
            self.num_known_boxes = num_known_boxes
            with open(known_names_file, 'r') as f:
                self.known_names = map(str.strip, f.readlines())
            self.num_known_classes = len(self.known_names)
            self.known_obj_pub = rospy.Publisher('known_objects', ObjectArray, queue_size=10)
            rospy.Service('detect_known_objects', DetectObjects, self.detect_known_objects)
            rospy.Service('get_names_known', GetNames, self.get_names_known)

        self.embed_ckpt_file = embed_ckpt_file
        if embed_ckpt_file is not None:
            self.embed_dim = embed_dim
            self.num_embed_boxes = num_embed_boxes
            self.embed_pub = rospy.Publisher('objects_embed', ObjectArray, queue_size=10)
            rospy.Service('detect_objects_embed', DetectObjects, self.detect_objects_embed)

        srv = Server(YoloDetectorConfig, self.config)

    def config(self, config, level):
        self.grid_h = config["grid_height"]
        self.grid_w = config["grid_width"]
        self.threshold = config["threshold"]
        self.input_shape = self.grid_h*32, self.grid_w*32
        return config

    def make_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.ph_x = tf.placeholder(tf.float32,
                                       shape=[None, None, None, 3])
            out = self.net(self.ph_x, num_classes=self.num_classes,
                           num_boxes=self.num_boxes, is_training=False)
            print out
            self.bbox_pred = out['bbox_pred']
            self.obj_prob = out['obj_prob']
            self.cls_score = out['cls_score']
            #if self.feature_tensor:
            #c = feature.get_shape()[3]
            #self.ph_followeeness_weights = tf.placeholder(tf.float32, shape=[1,1,c,self.num_boxes])
            #self.ph_followeeness_biases = tf.placeholder(tf.float32, shape=[self.num_boxes])
            #self.followeeness = tf.nn.conv2d(feature, self.ph_followeeness_weights, [1,1,1,1], 'SAME')
            #self.followeeness = tf.nn.bias_add(self.followeeness, self.ph_followeeness_biases)
            self.saver = tf.train.Saver()

        if self.known_ckpt_file is not None:
            self.graph_known = tf.Graph()
            with self.graph_known.as_default():
                self.ph_feat = tf.placeholder(tf.float32,
                                              shape=[None, None, None, 1024])
                c = (self.num_known_classes + 5) * self.num_known_boxes
                net = tf.contrib.slim.conv2d(self.ph_feat, c, [1,1],
                                             activation_fn=None, normalizer_fn=None)
                shape = tf.concat([tf.shape(net)[:-1],
                                   [self.num_known_boxes,
                                    self.num_known_classes+5]], 0)
                net = tf.reshape(net, shape)
                self.known_bbox_pred = net[:,:,:,:,:4]
                obj_score = net[:,:,:,:,4]
                self.known_obj_prob = tf.sigmoid(obj_score)
                self.known_cls_score = net[:,:,:,:,5:]
                self.saver_known = tf.train.Saver()
        else:
            self.graph_known = None

        if self.embed_ckpt_file is not None:
            self.graph_embed = tf.Graph()
            with self.graph_embed.as_default():
                self.ph_feat_embed = tf.placeholder(tf.float32,
                                                    shape=[None, None, None, 1024])
                c = (self.embed_dim + 5) * self.num_embed_boxes
                net = tf.contrib.slim.conv2d(self.ph_feat_embed, c, [1,1],
                                             activation_fn=None, normalizer_fn=None)
                shape = tf.concat([tf.shape(net)[:-1],
                                   [self.num_embed_boxes,
                                    self.embed_dim+5]], 0)
                net = tf.reshape(net, shape)
                self.embed_bbox_pred = net[:,:,:,:,:4]
                obj_score = net[:,:,:,:,4]
                self.embed_obj_prob = tf.sigmoid(obj_score)
                self.embed_embedding = net[:,:,:,:,5:]
                self.saver_embed = tf.train.Saver()
        else:
            self.graph_embed = None

    def initialize(self):
        if self.sess:
            self.sess.close()
        if self.sess_known:
            self.sess_known.close()
        if self.sess_embed:
            self.sess_embed.close()
        self.make_graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)
        self.saver.restore(self.sess, self.ckpt_file)
        rospy.loginfo('YOLO9000 network was restored from {}.'.format(self.ckpt_file))
        if self.known_ckpt_file is not None:
            self.sess_known = tf.Session(graph=self.graph_known, config=config)
            self.saver_known.restore(self.sess_known, self.known_ckpt_file)
            rospy.loginfo('Known objects layer was restored from {}.'.format(self.known_ckpt_file))
        if self.embed_ckpt_file is not None:
            self.sess_embed = tf.Session(graph=self.graph_embed, config=config)
            self.saver_embed.restore(self.sess_embed, self.embed_ckpt_file)
            rospy.loginfo('Embedding layer was restored from {}.'.format(self.embed_ckpt_file))

    def callback(self, data):
        num_connections = self.obj_pub.get_num_connections()
        if self.known_ckpt_file is not None:
            num_connections += self.known_obj_pub.get_num_connections()
        if self.embed_ckpt_file is not None:
            num_connections += self.embed_pub.get_num_connections()
        if num_connections == 0:
            return
        if not self.sess:
            return
            #self.initialize()
        result = self.process_imgmsg(data)
        if len(result) == 4:
            self.feat_pub.publish(result[3])
        self.obj_pub.publish(result[0])
        if self.known_ckpt_file is not None:
            self.known_obj_pub.publish(result[1])
        if self.embed_ckpt_file is not None:
            self.embed_pub.publish(result[2])

    def detect_objects(self, req):
        result = self.process_imgmsg(req.image)
        return DetectObjectsResponse(objects=result[0])

    def detect_known_objects(self, req):
        result = self.process_imgmsg(req.image)
        return DetectObjectsResponse(objects=result[1])

    def detect_objects_embed(self, req):
        result = self.process_imgmsg(req.image)
        return DetectObjectsResponse(objects=result[2])

    def _decode_predictions(self, imgmsg, bbox_pred, obj_prob, cls_score,
                            softmax_group_begin = None,
                            num_boxes = None,
                            feature_dim = None):
        if softmax_group_begin is None:
            softmax_group_begin = self.group_begin
        if num_boxes is None:
            num_boxes = self.num_boxes
        if feature_dim is None:
            feature_dim = self.num_classes

        xs = (np.arange(self.grid_w).reshape([1,-1,1]) + sigmoid(bbox_pred[0,:,:,:,0]))/self.grid_w
        ys = (np.arange(self.grid_h).reshape([-1,1,1]) + sigmoid(bbox_pred[0,:,:,:,1]))/self.grid_h
        ws = np.exp(bbox_pred[0,:,:,:,2])*self.anchor_ws.reshape([1,1,-1])/self.grid_w
        hs = np.exp(bbox_pred[0,:,:,:,3])*self.anchor_hs.reshape([1,1,-1])/self.grid_h
        xs = (xs-ws/2) * imgmsg.width
        ys = (ys-hs/2) * imgmsg.height
        ws = ws * imgmsg.width
        hs = hs * imgmsg.height
        box_inds = np.argsort(obj_prob.flat)[::-1]
        if softmax_group_begin is not None:
            softmax_tree(cls_score, softmax_group_begin)
        msg = ObjectArray()
        msg.header.stamp = imgmsg.header.stamp
        msg.header.frame_id = imgmsg.header.frame_id
        rospy.loginfo('Maximum objectness: {}'.format(obj_prob.flat[box_inds[0]]))
        for i in box_inds:
            if obj_prob.flat[i] < self.threshold:
                continue
            x = xs.flat[i]
            y = ys.flat[i]
            w = ws.flat[i]
            h = hs.flat[i]
            obj = ObjectDesc()
            obj.row = (i/num_boxes) / self.grid_w
            obj.column = (i/num_boxes) % self.grid_w
            obj.box_index = i % num_boxes
            obj.top = y
            obj.left = x
            obj.bottom = y+h
            obj.right = x+w
            obj.objectness = obj_prob.flat[i]
            obj.class_probability = cls_score.flat[i*feature_dim:(i+1)*feature_dim]
            msg.objects.append(obj)
        return msg

    def process_imgmsg(self, imgmsg):
        try:
            #from sensor_msgs.msg import Image
            #imgmsg = rospy.wait_for_message(imgmsg.header.frame_id, Image)
            cv_image = self.bridge.imgmsg_to_cv2(imgmsg, 'rgb8')
        except CvBridgeError as e:
            rospy.logerr(e)

        image_batch = np.expand_dims(cv2.resize(cv_image/255.,
                                                self.input_shape[::-1]), 0)

        fetch = [self.bbox_pred, self.obj_prob, self.cls_score, self.feature_tensor]
        predictions = self.sess.run(fetch, {self.ph_x: image_batch})
        bbox_pred, obj_prob, cls_score, feat = predictions
        msg = self._decode_predictions(imgmsg, bbox_pred, obj_prob, cls_score)

        if self.known_ckpt_file is not None:
            fetch = [self.known_bbox_pred, self.known_obj_prob, self.known_cls_score]
            predictions = self.sess_known.run(fetch, {self.ph_feat: feat})
            bbox_pred, obj_prob, cls_score = predictions
            msg_known = self._decode_predictions(imgmsg, bbox_pred, obj_prob, cls_score,
                                                 softmax_group_begin = [0,cls_score.shape[-1]],
                                                 num_boxes = self.num_known_boxes,
                                                 feature_dim = self.num_known_classes)
        else:
            msg_known = None

        if self.embed_ckpt_file is not None:
            fetch = [self.embed_bbox_pred, self.embed_obj_prob, self.embed_embedding]
            predictions = self.sess_embed.run(fetch, {self.ph_feat_embed: feat})
            bbox_pred, obj_prob, embedding = predictions
            msg_embed = self._decode_predictions(imgmsg, bbox_pred, obj_prob, embedding,
                                                 softmax_group_begin = None,
                                                 num_boxes = self.num_embed_boxes,
                                                 feature_dim = self.embed_dim)
        else:
            msg_embed = None
            
        ret = [msg, msg_known, msg_embed]

        if self.feature_tensor:
            c = feat.shape[3]

            feat_msg = FeatureArray()
            feat_msg.header.stamp = msg.header.stamp
            feat_msg.header.frame_id = msg.header.frame_id

            data = np.zeros(len(msg.objects)*c, dtype=np.float32)
            flags = np.zeros((self.grid_h, self.grid_w), dtype=np.bool)
            for i, obj in enumerate(msg.objects):
                if flags[obj.row, obj.column]:
                    continue
                f = Feature()
                f.row = obj.row
                f.column = obj.column
                f.data = feat[0, obj.row, obj.column, :]
                feat_msg.features.append(f)
                flags[obj.row, obj.column] = True
            ret.append(feat_msg)
        return ret

    def get_names(self, req):
        res = GetNamesResponse()
        res.names = self.names
        res.tree_nodes = [TreeNode(parent=p)\
                          for p in self.tree]
        for i, node in enumerate(res.tree_nodes):
            if node.parent >= 0:
                parent = res.tree_nodes[node.parent]
                parent.children.append(i)
        return res
    
    def get_names_known(self, req):
        res = GetNamesResponse()
        res.names = self.known_names
        res.tree_nodes = [TreeNode(parent=-1)\
                          for _ in xrange(self.num_known_classes)]
        return res

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt_9000', type=str,
                        help='ckpt file for yolo9000.')
    parser.add_argument('names_9000', type=str,
                        help='names file for yolo9000.')
    parser.add_argument('tree_9000', type=str,
                        help='tree file for yolo9000.')
    parser.add_argument('--ckpt_known', type=str,
                        help='ckpt file for known objects.')
    parser.add_argument('--names_known', type=str,
                        help='names file for known objects.')
    parser.add_argument('--ckpt_embed', type=str,
                        help='ckpt file for embedding.')
    args, _ = parser.parse_known_args()

    rospy.init_node('yolo_9000_known_embed')
    yd = YoloDetector(args.ckpt_9000, args.names_9000, args.tree_9000,
                      args.ckpt_known, args.names_known,
                      args.ckpt_embed,
                      input_shape=(256,256))
    yd.initialize()
    rospy.spin()
