#!/usr/bin/env python

import numpy as np
import tensorflow as tf
import struct
import sys

def write_float32(f, array):
    return f.write(struct.pack('f'*len(array), *array))

def write_int32(f, array):
    return f.write(struct.pack('i'*len(array), *array))

#shape: h,w,channels,filters
#in yolo.weights: filters,channels,h,w
def write_conv2d(f, weights, biases):
    #HWCN -> NCHW
    weights = weights.transpose([3,2,0,1])

    write_float32(f, biases)
    write_float32(f, weights.flat)

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print 'Usage: rosrun yolo_ros convert_ckpt_to_weights.py ckpt_file num_classes num_boxes weights_file'
        quit()
        
    reader = tf.saver.NewCheckpointReader(sys.argv[1])
    num_classes = int(sys.argv[2])
    num_boxes = int(sys.argv[3])
    v2s = reader.get_variable_to_shape_map()
    if 'Conv/weights' not in v2s or 'Conv/biases' not in v2s:
        print 'Invalid ckpt file.'
        quit()
    if v2s['Conv/weights'][3] % num_boxes != 0:
        print 'Invalid ckpt file.'
        quit()
    if v2s['Conv/weights'][3]/num_boxes - 5 != num_classes:
        print 'Invalid ckpt file.'
        quit()
    weights = reader.get_tensor('Conv/weights')
    biases = reader.get_tensor('Conv/biases')
    with open(sys.argv[4], 'wb') as f:
        write_int32(f, [0,0,0,0])
        write_conv2d(f, weights, biases)
    print 'Saved {}.'.format(sys.argv[4])
