#!/usr/bin/env python

import os
import time
import cv2
import numpy as np
import tensorflow as tf
from nets import yolo_v3, yolo9000, yolo_loss
from yolo_tf import box_iou
import traceback
#from tarinai import box_iou
slim = tf.contrib.slim

MIN_FG_SCALE = 0.3
MAX_FG_SCALE = 2.

YOLO9000_FEATURE_TENSORS = ['Conv_17/Leaky:0']
YOLO9000_WEIGHTS_TENSORS = ['Conv_18/weights:0']
YOLO9000_BIASES_TENSORS = ['Conv_18/biases:0']
YOLO9000_FEATURE_DIMS = [1024]
YOLO9000_NUM_CLASSES = 9418
YOLO9000_ANCHORS = [np.array([[0.77871, 1.14074],
                              [3.00525, 4.31277],
                              [9.22725, 9.61974]], dtype=np.float32) * 32]
YOLO9000_OBJ_SCALES = [4.]
YOLO9000_NOOBJ_SCALES = [1.]
YOLO9000_COORD_SCALES = [1.]
YOLO9000_CLASS_SCALES = [1.]

YOLOV3_FEATURE_TENSORS = ['Conv_73/Leaky:0', 'Conv_65/Leaky:0', 'Conv_57/Leaky:0']
YOLOV3_WEIGHTS_TENSORS = ['Conv_74/weights:0', 'Conv_66/weights:0', 'Conv_58/weights:0']
YOLOV3_BIASES_TENSORS = ['Conv_74/biases:0', 'Conv_66/biases:0', 'Conv_58/biases:0']
YOLOV3_FEATURE_DIMS = [256, 512, 1024]
YOLOV3_NUM_CLASSES = 80
YOLOV3_ANCHORS = [np.array([[10,13],[16,30],[33,23]], dtype=np.float32),
                  np.array([[30,61],[62,45],[59,119]], dtype=np.float32),
                  np.array([[116,90],[156,198],[373,326]], dtype=np.float32)]
YOLOV3_OBJ_SCALES = [4., 1., 1.]
YOLOV3_NOOBJ_SCALES = [1., 1., 1.]
YOLOV3_COORD_SCALES = [1., 1., 1.]
YOLOV3_CLASS_SCALES = [1., 1., 1.]

import concurrent.futures

def sigmoid(x):
    return 1/(1+np.exp(-x))

class ClassifierLayer:
    def __init__(self):
        self.num_yolo_classes = None
        self.feature_dim = None
        self.anchors = None
        
        # yolo graph/session
        self.yolo_cls_score = None
        self.yolo_feature = None
        self.yolo_weights = None
        self.yolo_biases = None

        # detector graph/session
        self.ph_feature = None
        self.ph_labels = None
        self.ph_obj_cond = None
        self.ph_noobj_cond = None
        self.ph_obj_bbox_t = None
        self.outputs = None
        self.loss = None
        self.weights = None
        self.biases = None

        self.obj_scale = 4
        self.noobj_scale = 1
        self.class_scale = 1
        self.coord_scale = 1

class YoloTrain:
    def __init__(self, ckpt_file, bg_dir, fg_dir,
                 batch_size = 5,
                 num_objs = 3,
                 learning_rate = 0.0001,
                 log_dir = '/tmp/yolo'):
        self.ckpt_file = ckpt_file
        self.fg_dir = fg_dir

        self._find_files(bg_dir, fg_dir)
        self.num_boxes = 3

        self.learning_rate = learning_rate
        #self.learning_rate = 0.0002
        self.hue = .1
        self.saturation = .75
        self.exposure = .75
        self.batch_size = batch_size
        self.num_objs = num_objs

        self.executor = concurrent.futures.ThreadPoolExecutor()
        self.image_batch_future = None

        self.yolo_sess = None
        self.detector_sess = None

        if ckpt_file.endswith('yolo9000.ckpt'):
            num_classes = YOLO9000_NUM_CLASSES
            feat_names = YOLO9000_FEATURE_TENSORS
            weights_names = YOLO9000_WEIGHTS_TENSORS
            biases_names = YOLO9000_BIASES_TENSORS
            feature_dims = YOLO9000_FEATURE_DIMS
            anchors = YOLO9000_ANCHORS
            obj_scales = YOLO9000_OBJ_SCALES
            noobj_scales = YOLO9000_NOOBJ_SCALES
            class_scales = YOLO9000_CLASS_SCALES
            coord_scales = YOLO9000_COORD_SCALES
            self.yolo_net = lambda x, **args: [yolo9000(x, **args)]
        elif ckpt_file.endswith('yolov3.ckpt'):
            num_classes = YOLOV3_NUM_CLASSES
            feat_names = YOLOV3_FEATURE_TENSORS
            weights_names = YOLOV3_WEIGHTS_TENSORS
            biases_names = YOLOV3_BIASES_TENSORS
            feature_dims = YOLOV3_FEATURE_DIMS
            anchors = YOLOV3_ANCHORS
            obj_scales = YOLOV3_OBJ_SCALES
            noobj_scales = YOLOV3_NOOBJ_SCALES
            class_scales = YOLOV3_CLASS_SCALES
            coord_scales = YOLOV3_COORD_SCALES
            self.yolo_net = yolo_v3
        self._create_graph(feat_names, weights_names, biases_names, feature_dims)
        for l, _anchors, obj, noobj, cls, coord in zip(self.classifier_layers,
                                                       anchors,
                                                       obj_scales,
                                                       noobj_scales,
                                                       class_scales,
                                                       coord_scales):
            l.num_yolo_classes = num_classes
            l.anchors = _anchors
            l.obj_scale = obj
            l.noobj_scale = noobj
            l.class_scale = cls
            l.coord_scale = coord
        self.writer = tf.summary.FileWriter(log_dir, self.detector_graph)

    def _find_files(self, bg_dir, fg_dir):
        isimg = lambda f: f.endswith('.png') or f.endswith('.jpg')

        self.bg_files = [os.path.join(bg_dir, f)
                         for f in os.listdir(bg_dir) if isimg(f)]
        self.fg_files = {}
        self.img_files = {}
        self.bg_dic = {}

        self.class_names = []
        #create forground image dic and class name
        for f in os.listdir(fg_dir):
            if not isimg(f):
                continue
            name = '.'.join(f.split('.')[:-1])
            name = '_'.join(name.split('_')[:-1])
            img_filename = os.path.join(fg_dir, f)
            if name not in self.class_names:
                self.class_names.append(name)
                self.fg_files[name] = []
            self.fg_files[name].append(img_filename)
            self.img_files[img_filename] = cv2.imread(img_filename, -1)
        #create background image dic
        for b in self.bg_files:
            self.bg_dic[b] = cv2.imread(b)
        self.num_classes = len(self.class_names)
        print self.class_names
        #print self.fg_files

    def _create_graph(self,
                      feature_tensor_names,
                      weights_tensor_names,
                      biases_tensor_names,
                      feature_dims):
        self.yolo_graph = tf.Graph()
        self.classifier_layers = []
        with self.yolo_graph.as_default():
            self.ph_image = tf.placeholder(tf.float32,
                                           shape=[None, None, None, 3])
            out = self.yolo_net(self.ph_image, is_training=False)
            assert len(out) == len(feature_tensor_names)
            for x, feat, weights, biases, dim in zip(out, feature_tensor_names,
                                                     weights_tensor_names, biases_tensor_names,
                                                     feature_dims):
                l = ClassifierLayer()
                l.yolo_cls_score = x['cls_score']
                l.yolo_feature = self.yolo_graph.get_tensor_by_name(feat)
                l.yolo_weights = self.yolo_graph.get_tensor_by_name(weights)
                l.yolo_biases = self.yolo_graph.get_tensor_by_name(biases)
                l.feature_dim = dim
                self.classifier_layers.append(l)
            self.yolo_saver = tf.train.Saver()

        self.detector_graph = tf.Graph()
        with self.detector_graph.as_default():
            total_loss = None
            summaries = []
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            c = (self.num_classes + 5) * self.num_boxes
            for i, l in enumerate(self.classifier_layers):
                l.ph_feature = tf.placeholder(tf.float32,
                                              shape=[None, None, None, l.feature_dim])
                l.ph_labels = tf.placeholder(tf.int32,
                                             shape=[None, None, None, self.num_boxes])
                l.ph_obj_cond = tf.placeholder(tf.bool,
                                               shape=[None, None, None, self.num_boxes])
                l.ph_noobj_cond = tf.placeholder(tf.bool,
                                                 shape=[None, None, None, self.num_boxes])
                l.ph_obj_bbox_t = tf.placeholder(tf.float32,
                                                 shape=[None, None, None, self.num_boxes, 4])

                scope = 'classifier_%d' % i
                net = slim.conv2d(l.ph_feature, c, [1,1],
                                  activation_fn=None, normalizer_fn=None,
                                  scope=scope)
                vs = self.detector_graph.get_collection('variables')
                ws = filter(lambda x: 'weights' in x.name and scope in x.name, vs)
                assert len(ws) == 1
                l.weights = ws[0]
                bs = filter(lambda x: 'biases' in x.name and scope in x.name, vs)
                assert len(bs) == 1
                l.biases = bs[0]
                
                shape = tf.concat([tf.shape(net)[:-1],
                                   [self.num_boxes, self.num_classes+5]], 0)
                net = tf.reshape(net, shape)
                bbox_pred = net[:,:,:,:,:4]
                obj_score = net[:,:,:,:,4]
                obj_prob = tf.sigmoid(obj_score)
                cls_score = net[:,:,:,:,5:]
                cls_prob = tf.nn.softmax(cls_score)
                l.outputs = {'bbox_pred': bbox_pred,
                             'obj_score': obj_score,
                             'obj_prob': obj_prob,
                             'cls_score': cls_score,
                             'cls_prob': cls_prob}
                summaries.append(tf.summary.histogram('classifier_%d/objectness' % i, obj_prob))
                summaries.append(tf.summary.histogram('classifier_%d/bbox_x' % i, tf.sigmoid(bbox_pred[:,:,:,:,0])))
                summaries.append(tf.summary.histogram('classifier_%d/bbox_y' % i, tf.sigmoid(bbox_pred[:,:,:,:,1])))
                summaries.append(tf.summary.histogram('classifier_%d/bbox_w' % i, tf.exp(bbox_pred[:,:,:,:,2])))
                summaries.append(tf.summary.histogram('classifier_%d/bbox_h' % i, tf.exp(bbox_pred[:,:,:,:,2])))

                losses = yolo_loss(l.outputs, l.ph_labels,
                                   l.ph_obj_cond, l.ph_noobj_cond,
                                   l.ph_obj_bbox_t)
                bbox_loss = tf.reduce_sum(tf.reduce_mean(losses['bbox_loss'], 0))
                cls_loss = tf.reduce_sum(tf.reduce_mean(losses['cls_loss'], 0))
                obj_loss = tf.reduce_sum(tf.reduce_mean(losses['obj_loss'], 0))
                noobj_loss = tf.reduce_sum(tf.reduce_mean(losses['noobj_loss'], 0))
                l.loss = bbox_loss * l.coord_scale\
                         + cls_loss * l.class_scale\
                         + obj_loss * l.obj_scale\
                         + noobj_loss * l.noobj_scale
                if total_loss is None:
                    total_loss = l.loss
                else:
                    total_loss = total_loss + l.loss
                summaries.append(tf.summary.scalar('classifier_%d/bbox_loss' % i, bbox_loss))
                summaries.append(tf.summary.scalar('classifier_%d/cls_loss' % i, cls_loss))
                summaries.append(tf.summary.scalar('classifier_%d/obj_loss' % i, obj_loss))
                summaries.append(tf.summary.scalar('classifier_%d/noobj_loss' % i, noobj_loss))
                summaries.append(tf.summary.scalar('classifier_%d/loss' % i, l.loss))
                
                l.ph_weights = tf.placeholder(tf.float32)
                l.ph_biases = tf.placeholder(tf.float32)
                l.assign_op = tf.group([l.weights.assign(l.ph_weights),
                                        l.biases.assign(l.ph_biases)])
            self.init_op = tf.global_variables_initializer()

            self.ph_detection_vis = tf.placeholder(tf.float32, shape=[None,None,None,3])
            summaries.append(tf.summary.image('detection', self.ph_detection_vis))
            summaries.append(tf.summary.scalar('loss', total_loss))
            self.summary = tf.summary.merge(summaries)
            self.train_op = opt.minimize(total_loss)
            self.detector_saver = tf.train.Saver()

    def initialize(self, ckpt=None):
        if self.yolo_sess:
            self.yolo_sess.close()
        self.yolo_sess = tf.Session(graph=self.yolo_graph)
        self.yolo_saver.restore(self.yolo_sess, self.ckpt_file)
        for l in self.classifier_layers:
            # convert tensors to values
            l.yolo_weights, l.yolo_biases = self.yolo_sess.run([l.yolo_weights, l.yolo_biases])

        print 'Network was restored from {}.'.format(self.ckpt_file)

        if self.detector_sess:
            self.detector_sess.close()
        self.detector_sess = tf.Session(graph=self.detector_graph)
        self.detector_sess.run(self.init_op)

        if ckpt is not None:
            reader = tf.train.NewCheckpointReader(ckpt)
            
        for k, l in enumerate(self.classifier_layers):
            if ckpt is not None:
                weights = reader.get_tensor('classifier_%d/weights' % k)
                biases = reader.get_tensor('classifier_%d/biases' % k)
            else:
                weights, biases = self.detector_sess.run([l.weights, l.biases])
            for i in range(self.num_boxes):
                offset = (5+self.num_classes) * i
                offset_9k = (5+l.num_yolo_classes) * i
                weights[:,:,:,offset:offset+5] = l.yolo_weights[:,:,:,offset_9k:offset_9k+5]
                biases[offset:offset+5] = l.yolo_biases[offset_9k:offset_9k+5]
            self.detector_sess.run(l.assign_op, {l.ph_weights: weights,
                                                 l.ph_biases: biases})
        self.image_batch_future = self.executor.submit(self.create_image_batch)

    def distort_image(self, img):
        h = np.random.rand() * self.hue*2 - self.hue
        s = np.random.rand() * abs(1.-self.saturation) + min(1., self.saturation)
        if np.random.rand() > .5:
            s = 1./s
        v = np.random.rand() * abs(1.-self.exposure) + min(1., self.exposure)
        if np.random.rand() > .5:
            v = 1./v
        hsv = cv2.cvtColor(img[:,:,:3], cv2.COLOR_BGR2HSV)/255.
        hscale = 179./255. # Hue is in [0,179] in OpenCV!
        hsv[:,:,0] = (hsv[:,:,0] + h*hscale) % hscale
        hsv[:,:,1] *= s
        hsv[np.where(hsv[:,:,1] > 1.)+(1,)] = 1.
        hsv[:,:,2] *= v
        hsv[np.where(hsv[:,:,2] > 1.)+(2,)] = 1.
        out = cv2.cvtColor(np.uint8(hsv*255.), cv2.COLOR_HSV2BGR)
        if img.shape[2] == 4:
            out = np.dstack([out, img[:,:,3]])
        return out

    def create_image_batch(self):
        scale = np.random.randint(10, 15)
        imsize = 32*scale

        img = np.zeros((self.batch_size, imsize, imsize, 3), dtype=np.float32)
        bbox = np.zeros((self.batch_size, self.num_objs, 4), dtype=np.float32)
        cls = np.zeros((self.batch_size, self.num_objs), dtype=np.int32)

        #for i in xrange(size):
        def create_image(i):
            bg_key = np.random.choice(self.bg_files)
            bg_img = self.distort_image(self.bg_dic[bg_key])/255.

            dw = bg_img.shape[1] - imsize
            dh = bg_img.shape[0] - imsize
            if dw <= 0 or dh <= 0:
                print 'Background image is too small!'
                return None
            left = np.random.randint(0, dw)
            right = left + imsize
            top = np.random.randint(0, dh)
            bottom = top + imsize
            img[i,:,:,:] = bg_img[top:bottom,left:right,::-1] #BGR -> RGB

            for j, cind in enumerate(np.random.choice(self.num_classes, self.num_objs)):
                cname = self.class_names[cind]
                fg_file = np.random.choice(self.fg_files[cname])
                #fg_img = cv2.imread(fg_file, -1)
                #key = np.random.choice(os.listdir(self.fg_dir))
                fg_img = self.img_files[fg_file]
                if fg_img.shape[2]==4:
                    fg_alpha = fg_img[:,:,3]
                    opaque_y, opaque_x = np.where(fg_alpha > 0)
                    left = np.min(opaque_x)
                    right = np.max(opaque_x)
                    top = np.min(opaque_y)
                    bottom = np.max(opaque_y)
                    fg_img = fg_img[top:(bottom+1),left:(right+1),:]
                fg_img = self.distort_image(fg_img)/255.
                min_scale = MIN_FG_SCALE
                max_scale = min(float(imsize)/fg_img.shape[0], float(imsize)/fg_img.shape[1])
                max_scale = min(max_scale, MAX_FG_SCALE)
                scale = np.random.uniform(min_scale, max_scale)
                fg_img = cv2.resize(fg_img, (int(fg_img.shape[1]*scale), int(fg_img.shape[0]*scale)))
                dw = imsize - fg_img.shape[1]
                dh = imsize - fg_img.shape[0]
                
                #if dw <= 0 or dh <= 0:
                #    scale = max(fg_img.shape[0]/float(imsize), fg_img.shape[1]/float(imsize))
                #    print 'Foreground image was resized.'
                #    fg_img = cv2.resize(fg_img, tuple(map(int, (fg_img.shape[1]/scale, fg_img.shape[0]/scale))))
                #    dw = imsize - fg_img.shape[1]
                #    dh = imsize - fg_img.shape[0]

                if fg_img.shape[2]==4:
                    fg_alpha = fg_img[:,:,3:]
                    fg_img = fg_img[:,:,:3]
                else:
                    fg_alpha = 1.
                fg_img = fg_img[:,:,::-1] #BGR -> RGB

                w = fg_img.shape[1]
                h = fg_img.shape[0]
                left = np.random.randint(0, dw+1)
                right = left + w
                top = np.random.randint(0, dh+1)
                bottom = top + h
                img[i,top:bottom,left:right,:] \
                    = img[i,top:bottom,left:right,:]*(1-fg_alpha) + fg_img*fg_alpha
                
                x = (left+right)/2./imsize
                y = (top+bottom)/2./imsize
                w = float(w)/imsize
                h = float(h)/imsize
                bbox[i,j,:] = [x,y,w,h]
                cls[i,j] = cind
        futures = [self.executor.submit(create_image, i) \
                   for i in xrange(self.batch_size)]
        concurrent.futures.wait(futures)
        #for i in xrange(self.batch_size):
        #    create_image(i)
        return img, bbox, cls

    def create_image_batch_concurrent(self):
        prev = self.image_batch_future
        if prev is None:
            prev = self.executor.submit(self.create_image_batch)
        self.image_batch_future = self.executor.submit(self.create_image_batch)
        return prev.result()

    def create_truth_batch(self, img, bbox, cls, bbox_pred, anchors):
        size, rows, cols = bbox_pred.shape[:3]
        num_objs = bbox.shape[1]
        n = anchors.shape[0]
        labels = np.zeros((size, rows, cols, n), dtype=np.float32)
        obj_cond = np.zeros((size, rows, cols, n), dtype=np.bool)
        noobj_cond = np.ones((size, rows, cols, n), dtype=np.bool)
        obj_bbox_t = np.zeros((size, rows, cols, n, 4), dtype=np.float32)

        #for i in xrange(size):
        def create_truth(i):
            for j in xrange(num_objs):
                x, y, w, h = bbox[i,j]
                c = cls[i,j]
                col = int(x*cols)
                row = int(y*rows)

                max_iou = 0.
                box = None
                for k in xrange(n):
                    px, py, pw, ph = bbox_pred[i,row,col,k]
                    px = (col + sigmoid(px))/cols
                    py = (row + sigmoid(py))/rows
                    pw = np.exp(pw)*anchors[k,0]/img.shape[2]
                    ph = np.exp(ph)*anchors[k,1]/img.shape[1]
                    iou = box_iou(x,y,w,h,px,py,pw,ph)
                    if iou > max_iou:
                        max_iou = iou
                        box = k

                tx = x*cols-col
                ty = y*rows-row
                if box is not None:
                    tw = np.log(w/anchors[box,0]*img.shape[2])
                    th = np.log(h/anchors[box,1]*img.shape[1])
                    obj_bbox_t[i,row,col,box,:] = [tx,ty,tw,th]
                    labels[i,row,col,box] = cls[i,j]
                    obj_cond[i,row,col,box] = True
                    noobj_cond[i,row,col,box] = False
                else:
                    for box in xrange(self.num_boxes):
                        tw = np.log(w/anchors[box,0]*img.shape[3])
                        th = np.log(h/anchors[box,1]*img.shape[2])
                        obj_bbox_t[i,row,col,box,:] = [tx,ty,tw,th]
                    labels[i,row,col,:] = cls[i,j]
                    obj_cond[i,row,col,:] = True
                    noobj_cond[i,row,col,:] = False
        futures = [self.executor.submit(create_truth, i) \
                   for i in xrange(size)]
        concurrent.futures.wait(futures)
        return labels, obj_cond, noobj_cond, obj_bbox_t

    def visualize_detection(self, img, objectness_list, cls_prob_list, bbox_pred_list):
      img = np.uint8(img[0,:,:,:]*255)
      for il, (l, objectness, cls_prob, bbox_pred) in enumerate(zip(self.classifier_layers,
                                                                    objectness_list,
                                                                    cls_prob_list,
                                                                    bbox_pred_list)):
        _, rows, cols, boxes = objectness.shape
        for i in xrange(rows):
          for j in xrange(cols):
            for k in xrange(boxes):
              if objectness[0,i,j,k] < .1:
                continue
              try:
                x,y,w,h = bbox_pred[0,i,j,k]
                x = (sigmoid(x)+j)/cols*img.shape[1]
                y = (sigmoid(y)+i)/rows*img.shape[0]
                w = np.exp(w)*l.anchors[k,0]
                h = np.exp(h)*l.anchors[k,1]
                left = (int)(x - w/2)
                top = (int)(y - h/2)
                right = (int)(left + w)
                bottom = (int)(top + h)
                left = max(min(left, img.shape[1]), 0)
                right = max(min(right, img.shape[1]), 0)
                top = max(min(top, img.shape[0]), 0)
                bottom = max(min(bottom, img.shape[0]), 0)
                c = cls_prob[0,i,j,k].argmax()
                p = cls_prob[0,i,j,k,c]
                if il%3 == 0:
                    cv2.rectangle(img, (left,top), (right,bottom),
                                  (255,0,0), thickness=2)
                elif il%3 == 1:
                    cv2.rectangle(img, (left,top), (right,bottom),
                                  (0,255,0), thickness=2)
                elif il%3 == 2:
                    cv2.rectangle(img, (left,top), (right,bottom),
                                  (0,0,255), thickness=2)
                cv2.putText(img, '{}: {}'.format(self.class_names[c], p),
                            (left, top), cv2.FONT_HERSHEY_PLAIN, 1., (255,0,255))
              except:
                print 'Failed to draw bounding box.'
                traceback.print_exc()
      return img

    def train(self, learn_from_yolo=False,
              write_summary=False,
              show_detection=False,
              step=None):
        print 'Creating image batch'
        batch = self.create_image_batch_concurrent()
        if batch is None:
            return np.nan
        img, bbox, cls = batch
        print 'Calculating feature'
        fetch_list = [l.yolo_feature for l in self.classifier_layers]
        fetch_list += [l.yolo_cls_score for l in self.classifier_layers]
        result = self.yolo_sess.run(fetch_list,
                                    {self.ph_image: img})
        feature_list = result[:len(self.classifier_layers)]
        yolo_cls_score_list = result[len(self.classifier_layers):]

        feed_dict = {l.ph_feature: feat for l, feat in zip(self.classifier_layers, feature_list)}
        fetch_list = [l.outputs['bbox_pred'] for l in self.classifier_layers]
        fetch_list += [l.outputs['obj_prob'] for l in self.classifier_layers]
        if learn_from_yolo:
            fetch_list += [l.outputs['cls_score'] for l in self.classifier_layers]
        if show_detection or write_summary:
            fetch_list += [l.outputs['cls_prob'] for l in self.classifier_layers]
        print 'Calculating output'
        result = self.detector_sess.run(fetch_list, feed_dict)
        if show_detection or write_summary:
            cls_prob_list = result[-len(self.classifier_layers):]
            result = result[:-len(self.classifier_layers)]
        if learn_from_yolo:
            cls_score_list = result[-len(self.classifier_layers):]
            result = result[:-len(self.classifier_layers)]
        bbox_pred_list = result[:len(self.classifier_layers)]
        objectness_list = result[len(self.classifier_layers):]

        if show_detection or write_summary:
            vis_img = self.visualize_detection(img, objectness_list, cls_prob_list, bbox_pred_list)

        print 'Creating truth batch'
        feed_dict = {}
        fetch_list = [self.train_op]
        for l, feature, bbox_pred in zip(self.classifier_layers,
                                         feature_list, bbox_pred_list):
            result = self.create_truth_batch(img, bbox, cls, bbox_pred, l.anchors)
            labels, obj_cond, noobj_cond, obj_bbox_t = result
            feed_dict.update({l.ph_feature: feature,
                              l.ph_labels: labels,
                              l.ph_obj_cond: obj_cond,
                              l.ph_noobj_cond: noobj_cond,
                              l.ph_obj_bbox_t: obj_bbox_t})
            fetch_list.append(l.loss)
        if write_summary:
            fetch_list.append(self.summary)
            feed_dict[self.ph_detection_vis] = np.expand_dims(vis_img, 0)
        print 'Updating'
        result = self.detector_sess.run(fetch_list, feed_dict)
        if write_summary:
            self.writer.add_summary(result[-1], step)
        
        if learn_from_yolo:
          print 'Transferring weights from yolo'
          size, rows, cols = bbox_pred.shape[:3]
          RATE = .5
          result = self.detector_sess.run([l.weights for l in self.classifier_layers]\
                                          + [l.biases for l in self.classifier_layers])
          weights_list = result[:len(self.classifier_layers)]
          biases_list = result[len(self.classifier_layers):]
          for l, cls_score, yolo_cls_score, weights, biases\
              in zip(self.classifier_layers,
                     cls_score_list, yolo_cls_score_list,
                     weights_list, biases_list):
            for b in range(size):
              for i in range(self.num_boxes):
                offset = (5 + l.num_yolo_classes) * i
                for (x, y, w, h), c in zip(bbox[b], cls[b]):
                  col = int(x*cols)
                  row = int(y*rows)
                  yolo_scores = yolo_cls_score[b,row,col,i,:]
                  yolo_cls = yolo_scores.argmax()

                  if yolo_scores[yolo_cls] > cls_score[b,row,col,i,c]:
                    #print 'Class "{}" learned from yolo class {}.'.format(self.class_names[c], yolo_cls)
                    yw = l.yolo_weights[:,0,0,offset+5+yolo_cls]
                    weights[:,0,0,c] *= 1-RATE
                    weights[:,0,0,c] += RATE*yw
                    yb = l.yolo_biases[offset+5+yolo_cls]
                    biases[c] *= 1-RATE
                    biases[c] += RATE*yb
            self.detector_sess.run(l.assign_op, {l.ph_weights: weights,
                                                 l.ph_biases: biases})

        if show_detection:
            cv2.imshow('Detection', vis_img[:,:,::-1])
            cv2.waitKey(1)

    def save(self, path, step=None):
        ckpt_out = path+'.ckpt'
        self.detector_saver.save(self.detector_sess, ckpt_out,
                                 global_step=step)
        
import sys
print 'Pretrained yolo ckpt file : ' + sys.argv[1]
print 'Background image dir : ' + sys.argv[2]
print 'Object image dir     : ' + sys.argv[3]
print 'Output file prefix   : ' + sys.argv[4]
if len(sys.argv) > 5:
    print 'ckpt file to be restored: ' + sys.argv[5]
    ckpt = sys.argv[5]
    if ckpt.strip('0123456789').endswith('-'):
        step_0 = int(ckpt.split('-')[-1])
    print 'Resuming training from step %d.' % step_0
else:
    ckpt = None
    step_0 = 0

train = YoloTrain(sys.argv[1], sys.argv[2], sys.argv[3],
                  learning_rate = 0.00005)
train.initialize(ckpt = ckpt)

with open(sys.argv[4]+'.names', 'w') as f:
    f.writelines(map(lambda s: s+'\n', train.class_names))

steps = 100000
for i in xrange(steps):
    step = i + step_0
    print 'step {}'.format(step)
    train.train(learn_from_yolo = step < 800,
                show_detection = False,
                write_summary = step%10==0,
                step=step)
    if step%1000 == 0:
        train.save(sys.argv[4], step)
