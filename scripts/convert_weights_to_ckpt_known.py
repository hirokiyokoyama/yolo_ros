#!/usr/bin/env python

import numpy as np
import tensorflow as tf
import struct
import gc
import argparse
import os

def read_float32(f, n):
    return np.array(struct.unpack('f'*n, f.read(4*n)))

def read_int32(f, n):
    return np.array(struct.unpack('i'*n, f.read(4*n)))

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

def load_known_weights(sess, n_in, n_out, filename):
    with open(filename, 'rb') as f:
        print 'major: {}'.format(read_int32(f, 1)[0])
        print 'minor: {}'.format(read_int32(f, 1)[0])
        print 'revision: {}'.format(read_int32(f, 1)[0])
        print 'seen: {}'.format(read_int32(f, 1)[0])

        assign_conv2d_weights(sess, f, 'Conv', [1,1,n_in,n_out], False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str,
                        help='Input ckpt file.')
    parser.add_argument('output', type=str,
                        help='Output ckpt file.')
    parser.add_argument('--classes', type=int, required=True,
                        help='Number of classes')
    parser.add_argument('--boxes', type=int, default=3,
                        help='Number of boxes in each grid')
    parser.add_argument('--dim', type=int, default=1024,
                        help='Dimension of feature map')
    args = parser.parse_args()

    ph_feat = tf.placeholder(tf.float32,
                             shape=[None, None, None, args.dim])
    c = (args.classes + 5) * args.boxes
    net = tf.contrib.slim.conv2d(ph_feat, c, [1,1],
                                 activation_fn=None,
                                 normalizer_fn=None)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        load_known_weights(sess, args.dim, c, args.input)
        print 'Loaded weights from {}.'.format(args.input)
        tf.train.Saver().save(sess, os.path.abspath(args.output))
        print 'Saved checkpoint {}.'.format(args.output)
