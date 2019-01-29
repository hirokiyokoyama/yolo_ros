#!/usr/bin/env python
# TODO: validation, control probability to choose same classes

import os
import time
import cv2
import numpy as np
import tensorflow as tf
from nets import yolo9000, yolo_loss
from yolo_ros import box_iou
#from tarinai import box_iou
slim = tf.contrib.slim

FEATURE_TENSOR = 'Conv_17/Leaky:0' #or maybe 'Conv_15/Leaky:0'
FEATURE_DIM = 1024
NUM_YOLO_CLASSES = 9418
TOP_CONV = 'Conv_18'

import concurrent.futures

def sigmoid(x):
    return 1/(1+np.exp(-x))

def combn2(n):
    return sum([[(i,j) for j in range(n)] for i in range(n)], [])

class Yolo9000Train:
    def __init__(self, ckpt_file, bg_dir, fg_dir,
                 embed_dim = 128, gamma = 1., same_class_prob = 0.3,
                 batch_size = 5,
                 num_objs = 2,
                 device='/cpu:0'):
        self.ckpt_file = ckpt_file
        self.device = device
        self.fg_dir = fg_dir

        self._find_files(bg_dir, fg_dir)
        self.num_boxes = 3
        self.anchor_ws = np.array([0.77871, 3.00525, 9.22725])
        self.anchor_hs = np.array([1.14074, 4.31277, 9.61974])

        self.obj_scale = 4
        self.noobj_scale = 1
        self.feature_scale = 1
        self.coord_scale = 1
        #self.learning_rate = 0.0001
        self.learning_rate = 0.0002
        self.hue = .05
        self.saturation = .75
        self.exposure = .75
        self.batch_size = batch_size
        self.num_objs = num_objs
        self.embed_dim = embed_dim
        self.gamma = gamma
        self.same_class_prob = same_class_prob

        self.executor = concurrent.futures.ThreadPoolExecutor()
        self.image_batch_future = None

        self.yolo9000_sess = None
        self.detector_sess = None
        self._create_graph()

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
        self.num_classes = len(self.class_names)
        #create background image dic
        for b in self.bg_files:
            self.bg_dic[b] = cv2.imread(b)

    def _create_graph(self):
        self.yolo9000_graph = tf.Graph()
        with self.yolo9000_graph.as_default():
          with tf.device(self.device):
            self.ph_image = tf.placeholder(tf.float32,
                                           shape=[None, None, None, 3])
            out = yolo9000(self.ph_image, is_training=False)
            self.yolo_cls_score = out['cls_score']
          self.yolo9000_saver = tf.train.Saver()
        self.feature = self.yolo9000_graph.get_tensor_by_name(FEATURE_TENSOR)

        self.detector_graph = tf.Graph()
        with self.detector_graph.as_default():
          with tf.device(self.device):
            c = (self.embed_dim + 5) * self.num_boxes
            self.ph_feature = tf.placeholder(tf.float32,
                                             shape=[None, None, None, FEATURE_DIM])
            self.ph_inds1 = tf.placeholder(tf.int32,
                                           shape=[None])
            self.ph_inds2 = tf.placeholder(tf.int32,
                                           shape=[None])
            self.ph_labels = tf.placeholder(tf.float32,
                                            shape=[None])
            self.ph_obj_cond = tf.placeholder(tf.bool,
                                              shape=[None, None, None, 3])
            self.ph_noobj_cond = tf.placeholder(tf.bool,
                                                shape=[None, None, None, 3])
            self.ph_obj_bbox_t = tf.placeholder(tf.float32,
                                                shape=[None, None, None, 3, 4])
            # train only this layer
            net = slim.conv2d(self.ph_feature, c, [1,1],
                              activation_fn=None, normalizer_fn=None)
            shape = tf.concat([tf.shape(net)[:-1],
                               [self.num_boxes, self.embed_dim+5]], 0)
            net = tf.reshape(net, shape)
            bbox_pred = net[:,:,:,:,:4]
            obj_score = net[:,:,:,:,4]
            obj_prob = tf.sigmoid(obj_score)
            cls_score = net[:,:,:,:,5:]
            cls_prob = tf.nn.softmax(cls_score)
            self.outputs = {'bbox_pred': bbox_pred,
                            'obj_score': obj_score,
                            'obj_prob': obj_prob,
                            'cls_score': cls_score,
                            'cls_prob': cls_prob}

            losses = yolo_loss(self.outputs, tf.cast(self.ph_obj_cond, tf.int32),
                               self.ph_obj_cond, self.ph_noobj_cond,
                               self.ph_obj_bbox_t)
            embs = tf.reshape(cls_score, [-1, self.embed_dim])
            embs1 = tf.nn.embedding_lookup(embs, self.ph_inds1)
            embs2 = tf.nn.embedding_lookup(embs, self.ph_inds2)
            dot = tf.reduce_sum(embs1*embs2, 1)
            norm1 = tf.reduce_sum(tf.square(embs1), 1)
            norm2 = tf.reduce_sum(tf.square(embs2), 1)
            cos = dot/tf.sqrt(norm1*norm2)
            match_prob = tf.pow((cos+1)/2, self.gamma)
            match_cross_entropy = self.ph_labels * tf.log(match_prob) \
                                  + (1-self.ph_labels) * tf.log(1-match_prob)
            feature_loss = tf.reduce_mean(match_cross_entropy)
            self.loss = losses['bbox_loss'] * self.coord_scale\
                        + losses['obj_loss'] * self.obj_scale\
                        + losses['noobj_loss'] * self.noobj_scale
            self.loss = tf.reduce_mean(self.loss) + self.feature_scale*feature_loss
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            self.init_op = tf.global_variables_initializer()
            self.train_op = opt.minimize(self.loss)

            vs = self.detector_graph.get_collection('variables')
            ws = filter(lambda x: 'weights' in x.name, vs)
            bs = filter(lambda x: 'biases' in x.name, vs)
            assert len(ws) == 1 and len(bs) == 1
            self.v_weights = ws[0]
            self.v_biases = bs[0]

            self.ph_weights = tf.placeholder(tf.float32)
            self.ph_biases = tf.placeholder(tf.float32)
            self.assign_ops = [self.v_weights.assign(self.ph_weights),
                               self.v_biases.assign(self.ph_biases)]
          self.detector_saver = tf.train.Saver()

    def initialize(self):
        if self.yolo9000_sess:
            self.yolo9000_sess.close()
        self.yolo9000_sess = tf.Session(graph=self.yolo9000_graph)
        self.yolo9000_saver.restore(self.yolo9000_sess, self.ckpt_file)
        self.yolo_weights, self.yolo_biases = self.yolo9000_sess.run([TOP_CONV+'/weights:0', TOP_CONV+'/biases:0'])

        print 'Network was restored from {}.'.format(self.ckpt_file)

        if self.detector_sess:
            self.detector_sess.close()
        self.detector_sess = tf.Session(graph=self.detector_graph)
        self.detector_sess.run(self.init_op)

        weights_val, biases_val = self.detector_sess.run([self.v_weights, self.v_biases])
        for i in range(self.num_boxes):
            offset = (5+self.embed_dim) * i
            offset_9k = (5+NUM_YOLO_CLASSES) * i
            weights_val[:,:,:,offset:offset+5] = self.yolo_weights[:,:,:,offset_9k:offset_9k+5]
            biases_val[offset:offset+5] = self.yolo_biases[offset_9k:offset_9k+5]
        self.detector_sess.run(self.assign_ops, {self.ph_weights: weights_val,
                                                 self.ph_biases: biases_val})
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

            has_same_class = np.random.uniform() < self.same_class_prob
            if has_same_class:
                class_inds = np.random.choice(self.num_classes, self.num_objs-1, replace=False)
                class_inds = np.concatenate([class_inds, np.random.choice(class_inds, 1)])
            else:
                class_inds = np.random.choice(self.num_classes, self.num_objs, replace=False)
                
            for j, cind in enumerate(class_inds):
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
                dw = imsize - fg_img.shape[1]
                dh = imsize - fg_img.shape[0]
                if dw <= 0 or dh <= 0:
                    scale = max(fg_img.shape[0]/float(imsize), fg_img.shape[1]/float(imsize))
                    print 'Foreground image was resized.'
                    fg_img = cv2.resize(fg_img, tuple(map(int, (fg_img.shape[1]/scale, fg_img.shape[0]/scale))))
                    dw = imsize - fg_img.shape[1]
                    dh = imsize - fg_img.shape[0]

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

    def create_truth_batch(self, bbox, cls, bbox_pred, cls_score):
        size, rows, cols = bbox_pred.shape[:3]
        num_objs = bbox.shape[1]
        assert num_objs==2, 'Currently only 2 objects per image is supported.'
        n = self.num_boxes
        #labels = np.zeros((size, rows, cols, n), dtype=np.float32)
        feature_loss = np.zeros((size), dtype=np.float32)
        obj_cond = np.zeros((size, rows, cols, n), dtype=np.bool)
        noobj_cond = np.ones((size, rows, cols, n), dtype=np.bool)
        obj_bbox_t = np.zeros((size, rows, cols, n, 4), dtype=np.float32)

        #for i in xrange(size):
        def create_truth(i):
            inds = []
            inds1 = []
            inds2 = []
            labels = []
            for j in xrange(num_objs):
                x, y, w, h = bbox[i,j]
                col = int(x*cols)
                row = int(y*rows)

                max_iou = 0.
                box = None
                for k in xrange(self.num_boxes):
                    px, py, pw, ph = bbox_pred[i,row,col,k]
                    px = (col + sigmoid(px))/cols
                    py = (row + sigmoid(py))/rows
                    pw = np.exp(pw)*self.anchor_ws[k]/cols
                    ph = np.exp(ph)*self.anchor_hs[k]/rows
                    iou = box_iou(x,y,w,h,px,py,pw,ph)
                    if iou > max_iou:
                        max_iou = iou
                        box = k

                tx = x*cols-col
                ty = y*rows-row
                if box is not None:
                    tw = np.log(w/self.anchor_ws[box]*cols)
                    th = np.log(h/self.anchor_hs[box]*rows)
                    obj_bbox_t[i,row,col,box,:] = [tx,ty,tw,th]
                    #labels[i,row,col,box] = cls[i,j]
                    obj_cond[i,row,col,box] = True
                    noobj_cond[i,row,col,box] = False
                    inds.append([i*rows*cols*n+row*cols*n+col*n+box])
                else:
                    for box in xrange(self.num_boxes):
                        tw = np.log(w/self.anchor_ws[box]*cols)
                        th = np.log(h/self.anchor_hs[box]*rows)
                        obj_bbox_t[i,row,col,box,:] = [tx,ty,tw,th]
                    #labels[i,row,col,:] = cls[i,j]
                    obj_cond[i,row,col,:] = True
                    noobj_cond[i,row,col,:] = False
                    inds.append([i*rows*cols*n+row*cols*n+col*n+box for box in range(self.num_boxes)])
                    
            for j,(k,l) in enumerate(combn2(num_objs)):
                label = float(cls[i,k] == cls[i,l])
                for ind1 in inds[k]:
                    for ind2 in inds[l]:
                        inds1.append(ind1)
                        inds2.append(ind2)
                        labels.append(label)
            return inds1, inds2, labels
        futures = [self.executor.submit(create_truth, i) \
                   for i in xrange(size)]
        inds1 = []
        inds2 = []
        labels = []
        for f in futures:
            result = f.result()
            inds1 = inds1 + result[0]
            inds2 = inds2 + result[1]
            labels = labels + result[2]
        return inds1, inds2, labels, obj_cond, noobj_cond, obj_bbox_t

    def train(self, show_detection=False):
        print 'Creating image batch'
        batch = self.create_image_batch_concurrent()
        if batch is None:
            return np.nan
        img, bbox, cls = batch
        print 'Calculating feature'
        feature = self.yolo9000_sess.run(self.feature,
                                         {self.ph_image: img})

        feed_dict = {self.ph_feature: feature}
        fetch_list = [self.outputs['bbox_pred'],
                      self.outputs['obj_prob'],
                      self.outputs['cls_score']]
        print 'Calculating output'
        out = self.detector_sess.run(fetch_list, feed_dict)
        bbox_pred, objectness, cls_score = out

        print 'Creating truth batch'
        result = self.create_truth_batch(bbox, cls, bbox_pred, cls_score)
        inds1, inds2, labels, obj_cond, noobj_cond, obj_bbox_t = result
        feed_dict = {self.ph_feature: feature,
                     self.ph_inds1: inds1,
                     self.ph_inds2: inds2,
                     self.ph_labels: labels,
                     self.ph_obj_cond: obj_cond,
                     self.ph_noobj_cond: noobj_cond,
                     self.ph_obj_bbox_t: obj_bbox_t}
        fetch_list = [self.loss, self.train_op]
        print 'Updating'
        loss, _ = self.detector_sess.run(fetch_list, feed_dict)

        if show_detection:
          img = np.uint8(img[0,:,:,:]*255)
          _, rows, cols, boxes = objectness.shape
          for i in xrange(rows):
            for j in xrange(cols):
              for k in xrange(boxes):
                if objectness[0,i,j,k] < .3:
                  continue
                x,y,w,h = bbox_pred[0,i,j,k]
                x = (sigmoid(x)+j)/cols*img.shape[1]
                y = (sigmoid(y)+i)/rows*img.shape[0]
                w = np.exp(w)*self.anchor_ws[k]/cols*img.shape[1]
                h = np.exp(h)*self.anchor_hs[k]/rows*img.shape[0]
                left = (int)(x - w/2)
                top = (int)(y - h/2)
                right = (int)(left + w)
                bottom = (int)(top + h)
                left = max(min(left, img.shape[1]), 0)
                right = max(min(right, img.shape[1]), 0)
                top = max(min(top, img.shape[0]), 0)
                bottom = max(min(bottom, img.shape[0]), 0)
                cv2.rectangle(img, (left,top), (right,bottom),
                                  (255,0,0), thickness=2)
          cv2.imshow('Detection', img[:,:,::-1])
          cv2.waitKey(10)
        return loss.mean()

    def save(self, path, step=None):
        ckpt_out = path+'.ckpt'
        self.detector_saver.save(self.detector_sess, ckpt_out,
                                 global_step=step)
        
import sys
print 'Pretrained ckpt file : ' + sys.argv[1]
print 'Background image dir : ' + sys.argv[2]
print 'Object image dir     : ' + sys.argv[3]
print 'Output file prefix   : ' + sys.argv[4]
device = '/cpu:0'
if len(sys.argv) > 5:
    device = sys.argv[5]
print 'Device               : ' + device

train = Yolo9000Train(sys.argv[1], sys.argv[2], sys.argv[3],
                      embed_dim = 128, gamma = 1.,
                      device = device)
train.initialize()

steps = 100000
for i in xrange(steps):
    print 'step {}'.format(i)
    loss = train.train(show_detection = i%10==0)
    print 'loss = {}'.format(loss)
    if i%1000 == 0:
        train.save(sys.argv[4], i)
