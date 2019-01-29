#!/usr/bin/env python

import numpy as np
import tensorflow as tf
import struct
from slim import slim
from nets import yolo, tiny_yolo_voc, yolo9000, yolo_v3
import sys
import gc

def read_float32(f, n):
    return np.array(struct.unpack('f'*n, f.read(4*n)))

def read_int32(f, n):
    return np.array(struct.unpack('i'*n, f.read(4*n)))

def read_int64(f, n):
    return np.array(struct.unpack('q'*n, f.read(8*n)))

#shape: h,w,channels,filters
#in yolo.weights: filters,channels,h,w
def load_conv2d_weights(f, shape, batch_norm):
    gc.collect()
    n = shape[3]
    ops = []
    feed_dict = {}
    if batch_norm:
        beta = read_float32(f, n)
        gamma = read_float32(f, n)
        moving_mean = read_float32(f, n)
        moving_variance = read_float32(f, n)
        ph_beta = tf.placeholder(tf.float32, shape=(n,))
        ph_gamma = tf.placeholder(tf.float32, shape=(n,))
        ph_moving_mean = tf.placeholder(tf.float32, shape=(n,))
        ph_moving_variance = tf.placeholder(tf.float32, shape=(n,))
        ops.append(tf.get_variable('BatchNorm/beta').assign(ph_beta))
        ops.append(tf.get_variable('BatchNorm/gamma').assign(ph_gamma))
        ops.append(tf.get_variable('BatchNorm/moving_mean').assign(ph_moving_mean))
        ops.append(tf.get_variable('BatchNorm/moving_variance').assign(ph_moving_variance))
        feed_dict[ph_beta] = beta
        feed_dict[ph_gamma] = gamma
        feed_dict[ph_moving_mean] = moving_mean
        feed_dict[ph_moving_variance] = moving_variance
    else:
        biases = read_float32(f, n)
        ph_biases = tf.placeholder(tf.float32, shape=(n,))
        ops.append(tf.get_variable('biases').assign(ph_biases))
        feed_dict[ph_biases] = biases
    weights = read_float32(f, np.prod(shape))
    #NCHW -> HWCN
    weights = weights.reshape(np.array(shape)[[3,2,0,1]]).transpose([2,3,1,0])
    ph_weights = tf.placeholder(tf.float32, shape=shape)
    ops.append(tf.get_variable('weights').assign(ph_weights))
    feed_dict[ph_weights] = weights
    return ops, feed_dict

def assign_conv2d_weights(sess, f, name, shape, batch_norm):
    print name
    with tf.variable_scope(name, reuse=True):
        ops, feed_dict = load_conv2d_weights(f, shape, batch_norm)
        sess.run(ops, feed_dict=feed_dict)

def load_yolo_weights(sess, filename):
    with open(filename, 'rb') as f:
        print 'major: {}'.format(read_int32(f, 1)[0])
        print 'minor: {}'.format(read_int32(f, 1)[0])
        print 'revision: {}'.format(read_int32(f, 1)[0])
        print 'seen: {}'.format(read_int32(f, 1)[0])

        assign_conv2d_weights(sess, f, 'Conv', [3,3,3,32], True)
        
        assign_conv2d_weights(sess, f, 'Conv_1', [3,3,32,64], True)
        
        assign_conv2d_weights(sess, f, 'Conv_2', [3,3,64,128], True)
        assign_conv2d_weights(sess, f, 'Conv_3', [1,1,128,64], True)
        assign_conv2d_weights(sess, f, 'Conv_4', [3,3,64,128], True)
        
        assign_conv2d_weights(sess, f, 'Conv_5', [3,3,128,256], True)
        assign_conv2d_weights(sess, f, 'Conv_6', [1,1,256,128], True)
        assign_conv2d_weights(sess, f, 'Conv_7', [3,3,128,256], True)
        
        assign_conv2d_weights(sess, f, 'Conv_8', [3,3,256,512], True)
        assign_conv2d_weights(sess, f, 'Conv_9', [1,1,512,256], True)
        assign_conv2d_weights(sess, f, 'Conv_10', [3,3,256,512], True)
        assign_conv2d_weights(sess, f, 'Conv_11', [1,1,512,256], True)
        assign_conv2d_weights(sess, f, 'Conv_12', [3,3,256,512], True)
        
        assign_conv2d_weights(sess, f, 'Conv_13', [3,3,512,1024], True)
        assign_conv2d_weights(sess, f, 'Conv_14', [1,1,1024,512], True)
        assign_conv2d_weights(sess, f, 'Conv_15', [3,3,512,1024], True)
        assign_conv2d_weights(sess, f, 'Conv_16', [1,1,1024,512], True)
        assign_conv2d_weights(sess, f, 'Conv_17', [3,3,512,1024], True)

        assign_conv2d_weights(sess, f, 'Conv_18', [3,3,1024,1024], True)
        assign_conv2d_weights(sess, f, 'Conv_19', [3,3,1024,1024], True)
        
        assign_conv2d_weights(sess, f, 'Conv_20', [3,3,1024+2048,1024], True)
        assign_conv2d_weights(sess, f, 'Conv_21', [1,1,1024,425], False)

def load_yolo9000_weights(sess, filename):
    with open(filename, 'rb') as f:
        print 'major: {}'.format(read_int32(f, 1)[0])
        print 'minor: {}'.format(read_int32(f, 1)[0])
        print 'revision: {}'.format(read_int32(f, 1)[0])
        print 'seen: {}'.format(read_int32(f, 1)[0])

        assign_conv2d_weights(sess, f, 'Conv', [3,3,3,32], True)
        
        assign_conv2d_weights(sess, f, 'Conv_1', [3,3,32,64], True)
        
        assign_conv2d_weights(sess, f, 'Conv_2', [3,3,64,128], True)
        assign_conv2d_weights(sess, f, 'Conv_3', [1,1,128,64], True)
        assign_conv2d_weights(sess, f, 'Conv_4', [3,3,64,128], True)
        
        assign_conv2d_weights(sess, f, 'Conv_5', [3,3,128,256], True)
        assign_conv2d_weights(sess, f, 'Conv_6', [1,1,256,128], True)
        assign_conv2d_weights(sess, f, 'Conv_7', [3,3,128,256], True)
        
        assign_conv2d_weights(sess, f, 'Conv_8', [3,3,256,512], True)
        assign_conv2d_weights(sess, f, 'Conv_9', [1,1,512,256], True)
        assign_conv2d_weights(sess, f, 'Conv_10', [3,3,256,512], True)
        assign_conv2d_weights(sess, f, 'Conv_11', [1,1,512,256], True)
        assign_conv2d_weights(sess, f, 'Conv_12', [3,3,256,512], True)
        
        assign_conv2d_weights(sess, f, 'Conv_13', [3,3,512,1024], True)
        assign_conv2d_weights(sess, f, 'Conv_14', [1,1,1024,512], True)
        assign_conv2d_weights(sess, f, 'Conv_15', [3,3,512,1024], True)
        assign_conv2d_weights(sess, f, 'Conv_16', [1,1,1024,512], True)
        assign_conv2d_weights(sess, f, 'Conv_17', [3,3,512,1024], True)

        assign_conv2d_weights(sess, f, 'Conv_18', [1,1,1024,28269], False)

def load_tiny_yolo_voc_weights(sess, filename):
    with open(filename, 'rb') as f:
        print 'major: {}'.format(read_int32(f, 1)[0])
        print 'minor: {}'.format(read_int32(f, 1)[0])
        print 'revision: {}'.format(read_int32(f, 1)[0])
        print 'seen: {}'.format(read_int32(f, 1)[0])

        assign_conv2d_weights(sess, f, 'Conv', [3,3,3,16], True)
        assign_conv2d_weights(sess, f, 'Conv_1', [3,3,16,32], True)
        assign_conv2d_weights(sess, f, 'Conv_2', [3,3,32,64], True)
        assign_conv2d_weights(sess, f, 'Conv_3', [3,3,64,128], True)
        assign_conv2d_weights(sess, f, 'Conv_4', [3,3,128,256], True)
        assign_conv2d_weights(sess, f, 'Conv_5', [3,3,256,512], True)
        assign_conv2d_weights(sess, f, 'Conv_6', [3,3,512,1024], True)

        assign_conv2d_weights(sess, f, 'Conv_7', [3,3,1024,1024], True)
        assign_conv2d_weights(sess, f, 'Conv_8', [1,1,1024,125], False)

def load_yolov3_weights(sess, filename):
    with open(filename, 'rb') as f:
        print 'major: {}'.format(read_int32(f, 1)[0])
        print 'minor: {}'.format(read_int32(f, 1)[0])
        print 'revision: {}'.format(read_int32(f, 1)[0])
        print 'seen: {}'.format(read_int64(f, 1)[0]) # perhaps 8 bytes!?

        assign_conv2d_weights(sess, f, 'Conv', [3,3,3,32], True)

        assign_conv2d_weights(sess, f, 'Conv_1', [3,3,32,64], True)
        assign_conv2d_weights(sess, f, 'Conv_2', [1,1,64,32], True)
        assign_conv2d_weights(sess, f, 'Conv_3', [3,3,32,64], True)
        
        assign_conv2d_weights(sess, f, 'Conv_4', [3,3,64,128], True)
        assign_conv2d_weights(sess, f, 'Conv_5', [1,1,128,64], True)
        assign_conv2d_weights(sess, f, 'Conv_6', [3,3,64,128], True)
        assign_conv2d_weights(sess, f, 'Conv_7', [1,1,128,64], True)
        assign_conv2d_weights(sess, f, 'Conv_8', [3,3,64,128], True)
        
        assign_conv2d_weights(sess, f, 'Conv_9', [3,3,128,256], True)
        assign_conv2d_weights(sess, f, 'Conv_10', [1,1,256,128], True)
        assign_conv2d_weights(sess, f, 'Conv_11', [3,3,128,256], True)
        assign_conv2d_weights(sess, f, 'Conv_12', [1,1,256,128], True)
        assign_conv2d_weights(sess, f, 'Conv_13', [3,3,128,256], True)
        assign_conv2d_weights(sess, f, 'Conv_14', [1,1,256,128], True)
        assign_conv2d_weights(sess, f, 'Conv_15', [3,3,128,256], True)
        assign_conv2d_weights(sess, f, 'Conv_16', [1,1,256,128], True)
        assign_conv2d_weights(sess, f, 'Conv_17', [3,3,128,256], True)
        assign_conv2d_weights(sess, f, 'Conv_18', [1,1,256,128], True)
        assign_conv2d_weights(sess, f, 'Conv_19', [3,3,128,256], True)
        assign_conv2d_weights(sess, f, 'Conv_20', [1,1,256,128], True)
        assign_conv2d_weights(sess, f, 'Conv_21', [3,3,128,256], True)
        assign_conv2d_weights(sess, f, 'Conv_22', [1,1,256,128], True)
        assign_conv2d_weights(sess, f, 'Conv_23', [3,3,128,256], True)
        assign_conv2d_weights(sess, f, 'Conv_24', [1,1,256,128], True)
        assign_conv2d_weights(sess, f, 'Conv_25', [3,3,128,256], True)
        
        assign_conv2d_weights(sess, f, 'Conv_26', [3,3,256,512], True)
        assign_conv2d_weights(sess, f, 'Conv_27', [1,1,512,256], True)
        assign_conv2d_weights(sess, f, 'Conv_28', [3,3,256,512], True)
        assign_conv2d_weights(sess, f, 'Conv_29', [1,1,512,256], True)
        assign_conv2d_weights(sess, f, 'Conv_30', [3,3,256,512], True)
        assign_conv2d_weights(sess, f, 'Conv_31', [1,1,512,256], True)
        assign_conv2d_weights(sess, f, 'Conv_32', [3,3,256,512], True)
        assign_conv2d_weights(sess, f, 'Conv_33', [1,1,512,256], True)
        assign_conv2d_weights(sess, f, 'Conv_34', [3,3,256,512], True)
        assign_conv2d_weights(sess, f, 'Conv_35', [1,1,512,256], True)
        assign_conv2d_weights(sess, f, 'Conv_36', [3,3,256,512], True)
        assign_conv2d_weights(sess, f, 'Conv_37', [1,1,512,256], True)
        assign_conv2d_weights(sess, f, 'Conv_38', [3,3,256,512], True)
        assign_conv2d_weights(sess, f, 'Conv_39', [1,1,512,256], True)
        assign_conv2d_weights(sess, f, 'Conv_40', [3,3,256,512], True)
        assign_conv2d_weights(sess, f, 'Conv_41', [1,1,512,256], True)
        assign_conv2d_weights(sess, f, 'Conv_42', [3,3,256,512], True)
        
        assign_conv2d_weights(sess, f, 'Conv_43', [3,3,512,1024], True)
        assign_conv2d_weights(sess, f, 'Conv_44', [1,1,1024,512], True)
        assign_conv2d_weights(sess, f, 'Conv_45', [3,3,512,1024], True)
        assign_conv2d_weights(sess, f, 'Conv_46', [1,1,1024,512], True)
        assign_conv2d_weights(sess, f, 'Conv_47', [3,3,512,1024], True)
        assign_conv2d_weights(sess, f, 'Conv_48', [1,1,1024,512], True)
        assign_conv2d_weights(sess, f, 'Conv_49', [3,3,512,1024], True)
        assign_conv2d_weights(sess, f, 'Conv_50', [1,1,1024,512], True)
        assign_conv2d_weights(sess, f, 'Conv_51', [3,3,512,1024], True)
        
        assign_conv2d_weights(sess, f, 'Conv_52', [1,1,1024,512], True)
        assign_conv2d_weights(sess, f, 'Conv_53', [3,3,512,1024], True)
        assign_conv2d_weights(sess, f, 'Conv_54', [1,1,1024,512], True)
        assign_conv2d_weights(sess, f, 'Conv_55', [3,3,512,1024], True)
        assign_conv2d_weights(sess, f, 'Conv_56', [1,1,1024,512], True)
        assign_conv2d_weights(sess, f, 'Conv_57', [3,3,512,1024], True)
        assign_conv2d_weights(sess, f, 'Conv_58', [1,1,1024,255], False)
        
        assign_conv2d_weights(sess, f, 'Conv_59', [1,1,512,256], True)
        assign_conv2d_weights(sess, f, 'Conv_60', [1,1,256+512,256], True)
        assign_conv2d_weights(sess, f, 'Conv_61', [3,3,256,512], True)
        assign_conv2d_weights(sess, f, 'Conv_62', [1,1,512,256], True)
        assign_conv2d_weights(sess, f, 'Conv_63', [3,3,256,512], True)
        assign_conv2d_weights(sess, f, 'Conv_64', [1,1,512,256], True)
        assign_conv2d_weights(sess, f, 'Conv_65', [3,3,256,512], True)
        assign_conv2d_weights(sess, f, 'Conv_66', [1,1,512,255], False)
        
        assign_conv2d_weights(sess, f, 'Conv_67', [1,1,256,128], True)
        assign_conv2d_weights(sess, f, 'Conv_68', [1,1,128+256,128], True)
        assign_conv2d_weights(sess, f, 'Conv_69', [3,3,128,256], True)
        assign_conv2d_weights(sess, f, 'Conv_70', [1,1,256,128], True)
        assign_conv2d_weights(sess, f, 'Conv_71', [3,3,128,256], True)
        assign_conv2d_weights(sess, f, 'Conv_72', [1,1,256,128], True)
        assign_conv2d_weights(sess, f, 'Conv_73', [3,3,128,256], True)
        assign_conv2d_weights(sess, f, 'Conv_74', [1,1,256,255], False)

if __name__ == '__main__':
    ph_x = tf.placeholder(tf.float32, shape=[None, 416, 416, 3])
    if sys.argv[1].endswith('yolo.weights'):
        print 'YOLO v2'
        out = yolo(ph_x, num_classes=80, num_boxes=5)
        load = load_yolo_weights
    elif sys.argv[1].endswith('tiny-yolo-voc.weights'):
        print 'Tiny YOLO VOC'
        out = tiny_yolo_voc(ph_x, num_classes=20, num_boxes=5)
        load = load_tiny_yolo_voc_weights
    elif sys.argv[1].endswith('yolo9000.weights'):
        print 'YOLO 9000'
        out = yolo9000(ph_x, num_classes=9418, num_boxes=3)
        load = load_yolo9000_weights
    elif sys.argv[1].endswith('yolov3.weights'):
        print 'YOLO v3'
        out = yolo_v3(ph_x, num_classes=80)
        load = load_yolov3_weights

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        load(sess, sys.argv[1])
        print 'Loaded weights from {}.'.format(sys.argv[1])
        tf.train.Saver().save(sess, sys.argv[2])
        print 'Saved checkpoint {}.'.format(sys.argv[2])
