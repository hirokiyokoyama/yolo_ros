#!/usr/bin/env python

import cv2
import numpy as np
import tensorflow as tf
import rospy
from sensor_msgs.msg import Image
from std_srvs.srv import Empty, EmptyResponse
from cv_bridge import CvBridge, CvBridgeError
from nets import yolo, tiny_yolo_voc, yolo9000, yolo_v3
from yolo_ros.msg import ObjectArray, ObjectDesc, TreeNode
from yolo_ros.srv import GetNames, GetNamesResponse
from yolo_ros.srv import DetectObjects, DetectObjectsResponse
from yolo_ros.cfg import YoloDetectorConfig

from dynamic_reconfigure.server import Server

def sigmoid(x):
    return 1/(1+np.exp(-x))

class BoxDecoder:
    def __init__(self, anchors):
        self.anchors = anchors.reshape([1,1,1,-1,2])

    def decode(self, data, im_wh):
        anchors = self.anchors.reshape([1,1,1,-1,2])
        
        # decode x and y
        grid_h = data.shape[1]
        grid_w = data.shape[2]
        data[:,:,:,:,:2] = sigmoid(data[:,:,:,:,:2])
        data[:,:,:,:,0] += np.arange(grid_w).reshape([1,1,-1,1])
        data[:,:,:,:,1] += np.arange(grid_h).reshape([1,-1,1,1])
        data[:,:,:,:,0] /= grid_w
        data[:,:,:,:,1] /= grid_h
        data[:,:,:,:,0] *= im_wh[0]
        data[:,:,:,:,1] *= im_wh[1]

        # decode w and h
        data[:,:,:,:,2:] = np.exp(data[:,:,:,:,2:]) * anchors

        # center to top-left
        data[:,:,:,:,:2] -= data[:,:,:,:,2:]/2

# subclasses must implement _make_graph
class Detector:
    def __init__(self, typ, ckpt_file):
        self.ckpt_file = ckpt_file
        self.typ = typ
        
        if typ == 'yolo':
            anchors = np.array([[0.738768, 2.42204, 4.30971, 10.246, 12.6868],
                                [0.874946, 2.65704, 7.04493, 4.59428, 11.8741]]).T
            anchors *= 32
            self.decoders = [BoxDecoder(anchors)]
            self.feature_dims = [1024]
            self.num_boxes = 3
        elif typ == 'tiny-yolo-voc':
            anchors = np.array([[1.08, 1.19],
                                [3.42, 4.41],
                                [6.63, 11.38],
                                [9.42, 5.11],
                                [16.62, 10.52]])
            anchors *= 32
            self.decoders = [BoxDecoder(anchors)]
            self.feature_dims = [1024]
            self.num_boxes = 3
        elif typ == 'yolo9000':
            anchors = np.array([[0.77871, 1.14074],
                                [3.00525, 4.31277],
                                [9.22725, 9.61974]])
            anchors *= 32
            self.decoders = [BoxDecoder(anchors)]
            self.feature_dims = [1024]
            self.num_boxes = 3
        elif typ == 'yolov3':
            anchors = np.array([[10,13],[16,30],[33,23],
                                [30,61],[62,45],[59,119],
                                [116,90],[156,198],[373,326]],
                               dtype=np.float32)
            self.decoders = [BoxDecoder(anchors[:3]),
                             BoxDecoder(anchors[3:6]),
                             BoxDecoder(anchors[6:])]
            self.feature_dims = [256, 512, 1024]
            self.num_boxes = 3
        else:
            raise ValueError('Unknown type: %s' % typ)
        
        self.sess = None
        self.pub = None

    def initialize(self):
        self.finalize()
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.ph_x, self.bbox_pred, self.obj_prob, self.cls_prob = self._make_graph()
            self.saver = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(graph=self.graph, config=config)
        self.saver.restore(sess, self.ckpt_file)
        rospy.loginfo('Network was restored from {}.'.format(self.ckpt_file))
        self.sess = sess

    def finalize(self):
        if self.sess:
            sess = self.sess
            self.sess = None
            sess.close()

    def process_data(self, data, orig_image, threshold, fetch_list=None):
        fetch = self.bbox_pred + self.obj_prob + self.cls_prob
        if fetch_list is not None:
            fetch += fetch_list
        msg = ObjectArray()
        if self.sess is None:
            return msg
        predictions = self.sess.run(fetch, {x: datum for x, datum in zip(self.ph_x, data)})
        n = len(self.bbox_pred)
        bbox_pred = predictions[:n]
        obj_prob = predictions[n:2*n]
        cls_prob = predictions[2*n:3*n]
        additional_outputs = predictions[3*n:]

        #rospy.loginfo('Maximum objectness: {}'.format(obj_prob.flat[box_inds[0]]))
        box_index = 0
        for l in xrange(n):
            bbox = bbox_pred[l]
            objp = obj_prob[l]
            clsp = cls_prob[l]
            self.decoders[l].decode(bbox, orig_image.shape[1::-1])
            
            for i in xrange(bbox.shape[1]):
                for j in xrange(bbox.shape[2]):
                    for k in xrange(bbox.shape[3]):
                        if objp[0,i,j,k] < threshold:
                            continue
                        x, y, w, h = bbox[0,i,j,k]
                        obj = ObjectDesc()
                        obj.row = i
                        obj.column = j
                        obj.box_index = box_index + k
                        obj.top = y
                        obj.left = x
                        obj.bottom = y+h
                        obj.right = x+w
                        obj.objectness = objp[0,i,j,k]
                        obj.class_probability = clsp[0,i,j,k]
                        msg.objects.append(obj)
            box_index += bbox.shape[3]

        if fetch_list is None:
            return msg
        return msg, additional_outputs

    def detect_objects(self, owner, req):
        result = owner.process_imgmsg(req.image, [self])
        return DetectObjectsResponse(objects=result[0])

class Classifier(Detector):
    def __init__(self, typ, ckpt_file, additional=False, names_file=None, tree_file=None):
        Detector.__init__(self, typ, ckpt_file)

        if additional:
            self.num_classes = None
            self.net = None
        else:
            if typ == 'yolo':
                self.num_classes = 80
                self.net = lambda x, **args: [yolo]
            elif typ == 'tiny-yolo-voc':
                self.num_classes = 20
                self.net = lambda x, **args: [tiny_yolo_voc]
            elif typ == 'yolo9000':
                self.num_classes = 9418
                self.net = lambda x, **args: [yolo9000(x, **args)]
            elif typ == 'yolov3':
                self.num_classes = 80
                self.net = yolo_v3

        if names_file is not None:
            with open(names_file, 'r') as f:
                self.names = map(str.strip, f.readlines())
            if self.num_classes is not None:
                assert len(self.names) == self.num_classes, 'Names file must have %d entries.' % self.num_classes
            else:
                self.num_classes = len(self.names)
        else:
            if self.num_classes is not None:
                self.names = ['class_%d' % i for i in xrange(self.num_classes)]
            else:
                raise ValueError('Names file must be specified.')
        if tree_file is not None:
            with open(tree_file, 'r') as f:
                self.tree = [int(l.split()[1]) for l in f.readlines()]
            assert len(self.tree) == self.num_classes, 'Tree file must have %d entries.' % self.num_classes
        else:
            self.tree = [-1] * self.num_classes
        
        self.group_begin = [0]
        begin = 0
        end = 0
        while begin < len(self.tree):
            end = begin+1
            while end < len(self.tree) and self.tree[end] == self.tree[begin]:
                end += 1
            begin = end
            self.group_begin.append(begin)
        
    def _make_graph(self):
        if self.net is not None:
            ph_x = [tf.placeholder(tf.float32, shape=[None, None, None, 3])]
            out = self.net(ph_x[0], num_classes=self.num_classes,
                           is_training=False)
            bbox_pred = [x['bbox_pred'] for x in out]
            obj_prob = [x['obj_prob'] for x in out]
            cls_score = [x['cls_score'] for x in out]
        else:
            ph_x = [tf.placeholder(tf.float32, shape=[None, None, None, c]) for c in self.feature_dims]
            c = (self.num_classes + 5) * self.num_boxes
            bbox_pred = []
            obj_prob = []
            cls_score = []
            for i, x in enumerate(ph_x):
                scope = 'classifier_%d' % i
                out = tf.contrib.slim.conv2d(x, c, [1,1],
                                             activation_fn=None, normalizer_fn=None,
                                             scope=scope)
                shape = tf.concat([tf.shape(out)[:-1],
                                   [self.num_boxes, self.num_classes+5]], 0)
                out = tf.reshape(out, shape)
                bbox_pred.append(out[:,:,:,:,:4])
                obj_prob.append(out[:,:,:,:,4])
                cls_score.append(out[:,:,:,:,5:])
                
        group_sizes = [self.group_begin[i+1] - self.group_begin[i] for i in xrange(len (self.group_begin)-1)]
            
        cls_prob = []
        for _cls_score in cls_score:
	    _cls_prob = tf.reshape(_cls_score,[-1,self.num_classes])
            _cls_probs = tf.split(_cls_prob, group_sizes, 1)
            _cls_probs = [tf.nn.softmax(c) for c in _cls_probs]
            _cls_prob = tf.concat(_cls_probs, 1)
            _cls_prob = tf.reshape(_cls_prob, tf.shape(_cls_score))
    	    cls_prob.append(_cls_prob)

        return ph_x, bbox_pred, obj_prob, cls_prob
            
    def get_names(self, req):
        res = GetNamesResponse()
        res.names = self.names
        res.tree_nodes = [TreeNode(parent=p) for p in self.tree]
        for i, node in enumerate(res.tree_nodes):
            if node.parent >= 0:
                parent = res.tree_nodes[node.parent]
                parent.children.append(i)
        return res
        
class YoloDetector:
    def __init__(self, ckpt_file, names_file=None, tree_file=None,
                 input_shape=(416,416)):
        self.ckpt_file = ckpt_file
        input_shape = np.array(input_shape)
        self.input_shape = input_shape

        self.detectors = []
        if ckpt_file.endswith('yolo.ckpt'):
            typ = 'yolo'
            self.feature_tensors = []
        elif ckpt_file.endswith('tiny-yolo-voc.ckpt'):
            typ = 'tiny-yolo-voc'
            self.feature_tensors = []
        elif ckpt_file.endswith('yolo9000.ckpt'):
            typ = 'yolo9000'
            self.feature_tensors = ['Conv_17/Leaky:0']
        elif ckpt_file.endswith('yolov3.ckpt'):
            typ = 'yolov3'
            self.feature_tensors = ['Conv_73/Leaky:0', 'Conv_65/Leaky:0', 'Conv_57/Leaky:0']
        self.add_classifier(typ, ckpt_file, False, names_file, tree_file)

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('image', Image, self.callback)

        self.threshold = 0.3
        Server(YoloDetectorConfig, self.config)

    def config(self, config, level):
        self.threshold = config["threshold"]
        # difficult to support yolov3
        #self.grid_h = config["grid_height"]
        #self.grid_w = config["grid_width"]
        #self.input_shape = self.grid_h*32, self.grid_w*32
        return config

    def add_classifier(self, typ, ckpt_file, additional=True, names_file=None, tree_file=None,
                       obj_topic = 'objects',
                       obj_srv = 'detect_objects',
                       names_srv = 'get_names'):
        det = Classifier(typ, ckpt_file, additional, names_file, tree_file)
        det.pub = rospy.Publisher(obj_topic, ObjectArray, queue_size=10)
        self.detectors.append(det)
        rospy.Service(obj_srv, DetectObjects, lambda req: det.detect_objects(self, req))
        rospy.Service(names_srv, GetNames, det.get_names)
        
    def initialize(self):
        for det in self.detectors:
            det.initialize()

    def finalize(self):
        for det in self.detectors:
            det.finalize()

    def callback(self, data):
        detectors = filter(lambda det: det.pub.get_num_connections()>0, self.detectors)
        if not detectors:
            return
        for det in detectors:
            if det.sess is None:
                return
        result = self.process_imgmsg(data, detectors)
        for det, res in zip(detectors, result):
            det.pub.publish(res)

    def process_imgmsg(self, imgmsg, detectors):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(imgmsg, 'rgb8')
        except CvBridgeError as e:
            rospy.logerr(e)

        image_batch = np.expand_dims(cv2.resize(cv_image/255.,
                                                tuple(self.input_shape[::-1])), 0)
        msg0, data = self.detectors[0].process_data([image_batch], cv_image,
                                                    self.threshold, fetch_list=self.feature_tensors)
        msgs = []
        for det in detectors:
            if det is self.detectors[0]:
                msgs.append(msg0)
            else:
                msgs.append(det.process_data(data, cv_image, self.threshold))
        for msg in msgs:
            msg.header.stamp = imgmsg.header.stamp
            msg.header.frame_id = imgmsg.header.frame_id
        return msgs

def enable_yolo_detector(req):
    yd.initialize()
    return EmptyResponse()

def disable_yolo_detector(req):
    yd.finalize()
    return EmptyResponse()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, help='ckpt file.')
    parser.add_argument('--names', type=str, help='names file.')
    parser.add_argument('--tree', type=str, help='tree file.')
    parser.add_argument('--ckpt1', type=str, help='ckpt file for additional detector.')
    parser.add_argument('--names1', type=str, help='names file for additional detector.')
    parser.add_argument('--tree1', type=str, help='tree file for additional detector.')
    parser.add_argument('--type1', type=str, help='type of additinal detector (currently only classifier).')
    parser.add_argument('--ckpt2', type=str, help='ckpt file for additional detector.')
    parser.add_argument('--names2', type=str, help='names file for additional detector.')
    parser.add_argument('--tree2', type=str, help='tree file for additional detector.')
    parser.add_argument('--type2', type=str, help='type of additinal detector (currently only classifier).')
    args, _ = parser.parse_known_args()

    rospy.init_node('yolo')
    yd = YoloDetector(args.ckpt, args.names, args.tree)
    for typ, ckpt, names, tree in [(args.type1, args.ckpt1, args.names1, args.tree1),
                                   (args.type2, args.ckpt2, args.names2, args.tree2)]:
        if typ is None and ckpt is None and names is None and tree is None:
            continue
        if typ is None:
            print 'Type is not specified.'
            quit()
        if typ not in ['classifier']:
            print 'Unknown type %s.' % typ
            quit()
        if ckpt is None:
            print 'ckpt file is not specified.'
            quit()
        if typ == 'classifier':
            _typ = yd.detectors[0].typ
            yd.add_classifier(_typ, ckpt, True, names, tree,
                              obj_topic = 'known_objects',
                              obj_srv = 'detect_known_objects',
                              names_srv = 'get_names_known')

    rospy.Service('enable_yolo_detector', Empty, enable_yolo_detector)
    rospy.Service('disable_yolo_detector', Empty, disable_yolo_detector)
    yd.initialize()
    rospy.spin()
