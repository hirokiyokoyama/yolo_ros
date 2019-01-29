import tensorflow as tf
slim = tf.contrib.slim

def leaky(x):
    return tf.where(x > 0, x, .1*x, name='Leaky')

def reorg(x, stride, reverse):
    shape1 = tf.shape(x)
    b = shape1[0]
    h = shape1[1]
    w = shape1[2]
    c = shape1[3]
    if reverse:
        shape2 = tf.stack([b,h,w,stride,c/stride])
        shape3 = tf.stack([b,h*stride,w*stride,c/(stride*stride)])
    else:
        shape2 = tf.stack([b,h/stride,stride,w/stride,c*stride])
        shape3 = tf.stack([b,h/stride,w/stride,c*(stride*stride)])
    out = tf.reshape(x, shape2)
    out = tf.transpose(out, [0,1,3,2,4])
    out = tf.reshape(out, shape3)

    shape = x.get_shape().as_list()
    if shape[1] is not None:
        shape[1] = shape[1]*stride if reverse else shape[1]/stride
    if shape[2] is not None:
        shape[2] = shape[2]*stride if reverse else shape[2]/stride
    if shape[3] is not None:
        shape[3] = shape[3]/(stride*stride) if reverse else shape[3]*(stride*stride)
    out.set_shape(shape)
    return out

def upsample(x, stride):
    with tf.name_scope('Upsample'):
        orig_shape = tf.shape(x)
        x = tf.expand_dims(tf.expand_dims(x, 3), 2)
        x = tf.tile(x, [1,1,stride,1,stride,1])
        shape = tf.concat([orig_shape[0:1],
                           orig_shape[1:3] * stride,
                           orig_shape[3:4]], 0)
        return tf.reshape(x, shape)

def yolo(x, num_classes=80, num_boxes=5, is_training=True):
    c = (num_classes + 5) * num_boxes
    row = x.get_shape()[1].value
    col = x.get_shape()[2].value
    row = row/32 if row is not None else tf.shape(x)[1]/32
    col = col/32 if col is not None else tf.shape(x)[2]/32
    batch_norm_params = {'decay': 0.0005, 'epsilon': 0.00001,
                         'center': True, 'scale': True}
    with slim.arg_scope([slim.conv2d], stride=1, padding='SAME',
                        activation_fn=leaky, normalizer_fn=slim.batch_norm):
     with slim.arg_scope([slim.batch_norm], is_training=is_training, **batch_norm_params):
      with slim.arg_scope([slim.max_pool2d], stride=2):
        net = slim.conv2d(x, 32, [3,3])
        net = slim.max_pool2d(net, [2,2])
        
        net = slim.conv2d(net, 64, [3,3])
        net = slim.max_pool2d(net, [2,2])
        
        net = slim.conv2d(net, 128, [3,3])
        net = slim.conv2d(net, 64, [1,1])
        net = slim.conv2d(net, 128, [3,3])
        net = slim.max_pool2d(net, [2,2])
        
        net = slim.conv2d(net, 256, [3,3])
        net = slim.conv2d(net, 128, [1,1])
        net = slim.conv2d(net, 256, [3,3])
        net = slim.max_pool2d(net, [2,2])

        net = slim.conv2d(net, 512, [3,3])
        net = slim.conv2d(net, 256, [1,1])
        net = slim.conv2d(net, 512, [3,3])
        net = slim.conv2d(net, 256, [1,1])
        net = slim.conv2d(net, 512, [3,3])
        branch = reorg(net, 2, False)
        net = slim.max_pool2d(net, [2,2])

        net = slim.conv2d(net, 1024, [3,3])
        net = slim.conv2d(net, 512, [1,1])
        net = slim.conv2d(net, 1024, [3,3])
        net = slim.conv2d(net, 512, [1,1])
        net = slim.conv2d(net, 1024, [3,3])

        net = slim.conv2d(net, 1024, [3,3])
        net = slim.conv2d(net, 1024, [3,3])
        net = tf.concat(axis=3, values=[branch, net])

        net = slim.conv2d(net, 1024, [3,3])
        net = slim.conv2d(net, c, [1,1],
                          activation_fn=None, normalizer_fn=None)

    net = tf.reshape(net, [-1, row, col, num_boxes, num_classes+5])
    bbox_pred = net[:,:,:,:,:4]
    obj_score = net[:,:,:,:,4]
    obj_prob = tf.sigmoid(obj_score)
    cls_score = net[:,:,:,:,5:]
    cls_prob = tf.reshape(cls_score,[-1,num_classes])
    cls_prob = tf.nn.softmax(cls_prob)
    cls_prob = tf.reshape(cls_prob,[-1, row, col, num_boxes, num_classes])
    
    return {'bbox_pred':bbox_pred,
            'obj_score':obj_score, 'obj_prob':obj_prob,
            'cls_score':cls_score, 'cls_prob':cls_prob}

def yolo9000(x, group_sizes=None, num_classes=9418, num_boxes=3, is_training=True):
    c = (num_classes + 5) * num_boxes
    row = x.get_shape()[1].value
    col = x.get_shape()[2].value
    row = row/32 if row is not None else tf.shape(x)[1]/32
    col = col/32 if col is not None else tf.shape(x)[2]/32
    batch_norm_params = {'decay': 0.0005, 'epsilon': 0.00001,
                         'center': True, 'scale': True}
    with slim.arg_scope([slim.conv2d], stride=1, padding='SAME',
                        activation_fn=leaky, normalizer_fn=slim.batch_norm):
     with slim.arg_scope([slim.batch_norm], is_training=is_training, **batch_norm_params):
      with slim.arg_scope([slim.max_pool2d], stride=2):
        net = slim.conv2d(x, 32, [3,3])
        net = slim.max_pool2d(net, [2,2])
        
        net = slim.conv2d(net, 64, [3,3])
        net = slim.max_pool2d(net, [2,2])
        
        net = slim.conv2d(net, 128, [3,3])
        net = slim.conv2d(net, 64, [1,1])
        net = slim.conv2d(net, 128, [3,3])
        net = slim.max_pool2d(net, [2,2])
        
        net = slim.conv2d(net, 256, [3,3])
        net = slim.conv2d(net, 128, [1,1])
        net = slim.conv2d(net, 256, [3,3])
        net = slim.max_pool2d(net, [2,2])

        net = slim.conv2d(net, 512, [3,3])
        net = slim.conv2d(net, 256, [1,1])
        net = slim.conv2d(net, 512, [3,3])
        net = slim.conv2d(net, 256, [1,1])
        net = slim.conv2d(net, 512, [3,3])
        net = slim.max_pool2d(net, [2,2])

        net = slim.conv2d(net, 1024, [3,3])
        net = slim.conv2d(net, 512, [1,1])
        net = slim.conv2d(net, 1024, [3,3])
        net = slim.conv2d(net, 512, [1,1])
        net = slim.conv2d(net, 1024, [3,3])

        net = slim.conv2d(net, c, [1,1],
                          activation_fn=None, normalizer_fn=None)

    net = tf.reshape(net, [-1, row, col, num_boxes, num_classes+5])
    bbox_pred = net[:,:,:,:,:4]
    obj_score = net[:,:,:,:,4]
    obj_prob = tf.sigmoid(obj_score)
    cls_score = net[:,:,:,:,5:]
    if group_sizes is None:
        return {'bbox_pred':bbox_pred, 'obj_score':obj_score,
                'obj_prob':obj_prob, 'cls_score':cls_score}
    cls_prob = tf.reshape(cls_score,[-1,num_classes])
    cls_probs = tf.split(cls_prob, group_sizes, 1)
    cls_probs = [tf.nn.softmax(c) for c in cls_probs]
    cls_prob = tf.concat(cls_probs, 1)
    cls_prob = tf.reshape(cls_prob,[-1, row, col, num_boxes, num_classes])
    
    return {'bbox_pred':bbox_pred,
            'obj_score':obj_score, 'obj_prob':obj_prob,
            'cls_score':cls_score, 'cls_prob':cls_prob}

def tiny_yolo_voc(x, num_classes=20, num_boxes=5, is_training=True):
    c = (num_classes + 5) * num_boxes
    row = x.get_shape()[1].value
    col = x.get_shape()[2].value
    row = row/32 if row is not None else tf.shape(x)[1]/32
    col = col/32 if col is not None else tf.shape(x)[2]/32
    batch_norm_params = {'decay': 0.0005, 'epsilon': 0.00001,
                         'center': True, 'scale': True}
    end_points = {}
    with slim.arg_scope([slim.conv2d], stride=1, padding='SAME',
                        activation_fn=leaky, normalizer_fn=slim.batch_norm):
     with slim.arg_scope([slim.batch_norm], is_training=is_training, **batch_norm_params):
      with slim.arg_scope([slim.max_pool2d], stride=2):
        net = slim.conv2d(x, 16, [3,3])
        end_points['Conv1'] = net
        net = slim.max_pool2d(net, [2,2])
        end_points['MaxPool1'] = net
        net = slim.conv2d(net, 32, [3,3])
        end_points['Conv2'] = net
        net = slim.max_pool2d(net, [2,2])
        end_points['MaxPool2'] = net
        net = slim.conv2d(net, 64, [3,3])
        end_points['Conv3'] = net
        net = slim.max_pool2d(net, [2,2])
        end_points['MaxPool3'] = net
        net = slim.conv2d(net, 128, [3,3])
        end_points['Conv4'] = net
        net = slim.max_pool2d(net, [2,2])
        end_points['MaxPool4'] = net
        net = slim.conv2d(net, 256, [3,3])
        end_points['Conv5'] = net
        net = slim.max_pool2d(net, [2,2])
        end_points['MaxPool5'] = net
        net = slim.conv2d(net, 512, [3,3])
        end_points['Conv6'] = net
        net = slim.max_pool2d(net, [2,2], stride=1, padding='SAME')
        end_points['MaxPool6'] = net
        net = slim.conv2d(net, 1024, [3,3])
        end_points['Conv7'] = net

        net = slim.conv2d(net, 1024, [3,3])
        end_points['Conv8'] = net
        net = slim.conv2d(net, c, [1,1],
                          activation_fn=None, normalizer_fn=None)
        end_points['Conv9'] = net
        
    net = tf.reshape(net, [-1, row, col, num_boxes, num_classes+5])
    bbox_pred = net[:,:,:,:,:4]
    end_points['bbox_pred'] = bbox_pred
    obj_score = net[:,:,:,:,4]
    end_points['obj_score'] = obj_score
    obj_prob = tf.sigmoid(obj_score)
    end_points['obj_prob'] = obj_prob
    cls_score = net[:,:,:,:,5:]
    end_points['cls_score'] = cls_score
    cls_prob = tf.reshape(cls_score,[-1,num_classes])
    cls_prob = tf.nn.softmax(cls_prob)
    cls_prob = tf.reshape(cls_prob,[-1, row, col, num_boxes, num_classes])
    end_points['cls_prob'] = cls_prob
    
    return end_points

def yolo_v3(x, num_classes=80, is_training=True):
    c = (num_classes + 5) * 3
    batch_norm_params = {'decay': 0.0005, 'epsilon': 0.00001,
                         'center': True, 'scale': True}
    with slim.arg_scope([slim.conv2d], stride=1, padding='SAME',
                        activation_fn=leaky, normalizer_fn=slim.batch_norm):
      with slim.arg_scope([slim.batch_norm], is_training=is_training, **batch_norm_params):
        paddings = tf.constant([[0,0],[1,0],[1,0],[0,0]])
        layers = []
        layers.append(slim.conv2d(x, 32, [3,3]))

        # Downsample (layers[1:])
        layers.append(slim.conv2d(tf.pad(layers[-1], paddings),
                                  64, [3,3], stride=2, padding='VALID'))
        layers.append(slim.conv2d(layers[-1], 32, [1,1]))
        layers.append(slim.conv2d(layers[-1], 64, [3,3]))
        layers.append(layers[-1] + layers[-3])
        
        # Downsample (layers[5:])
        layers.append(slim.conv2d(tf.pad(layers[-1], paddings),
                                  128, [3,3], stride=2, padding='VALID'))
        for _ in range(2):
            layers.append(slim.conv2d(layers[-1], 64, [1,1]))
            layers.append(slim.conv2d(layers[-1], 128, [3,3]))
            layers.append(layers[-1] + layers[-3])

        # Downsample (layers[12:])
        layers.append(slim.conv2d(tf.pad(layers[-1], paddings),
                                  256, [3,3], stride=2, padding='VALID'))
        for _ in range(8):
            layers.append(slim.conv2d(layers[-1], 128, [1,1]))
            layers.append(slim.conv2d(layers[-1], 256, [3,3]))
            layers.append(layers[-1] + layers[-3])

        # Downsample (layers[37:])
        layers.append(slim.conv2d(tf.pad(layers[-1], paddings),
                                  512, [3,3], stride=2, padding='VALID'))
        for _ in range(8):
            layers.append(slim.conv2d(layers[-1], 256, [1,1]))
            layers.append(slim.conv2d(layers[-1], 512, [3,3]))
            layers.append(layers[-1] + layers[-3])

        # Downsample (layers[62:])
        layers.append(slim.conv2d(tf.pad(layers[-1], paddings),
                                  1024, [3,3], stride=2, padding='VALID'))
        for _ in range(4):
            layers.append(slim.conv2d(layers[-1], 512, [1,1]))
            layers.append(slim.conv2d(layers[-1], 1024, [3,3]))
            layers.append(layers[-1] + layers[-3])

        ############# (layers[75:])
        for _ in range(3):
            layers.append(slim.conv2d(layers[-1], 512, [1,1]))
            layers.append(slim.conv2d(layers[-1], 1024, [3,3]))
        layers.append(slim.conv2d(layers[-1], c, [1,1],
                                  activation_fn = None,
                                  normalizer_fn = None))
        layers.append(layers[-1])
        yolo_678 = layers[-1]

        layers.append(layers[-4])
        layers.append(slim.conv2d(layers[-1], 256, [1,1]))
        layers.append(upsample(layers[-1], 2))
        layers.append(tf.concat([layers[-1], layers[61]], 3))
        for _ in range(3):
            layers.append(slim.conv2d(layers[-1], 256, [1,1]))
            layers.append(slim.conv2d(layers[-1], 512, [3,3]))
        layers.append(slim.conv2d(layers[-1], c, [1,1],
                                  activation_fn = None,
                                  normalizer_fn = None))
        layers.append(layers[-1])
        yolo_345 = layers[-1]

        layers.append(layers[-4])
        layers.append(slim.conv2d(layers[-1], 128, [1,1]))
        layers.append(upsample(layers[-1], 2))
        layers.append(tf.concat([layers[-1], layers[36]], 3))
        for _ in range(3):
            layers.append(slim.conv2d(layers[-1], 128, [1,1]))
            layers.append(slim.conv2d(layers[-1], 256, [3,3]))
        layers.append(slim.conv2d(layers[-1], c, [1,1],
                                  activation_fn = None,
                                  normalizer_fn = None))
        layers.append(layers[-1])
        yolo_012 = layers[-1]

    yolo_layers = []
    for net in [yolo_012, yolo_345, yolo_678]:
        orig_shape = tf.shape(net)
        shape = tf.concat([orig_shape[:3], [3, num_classes+5]], 0)
        net = tf.reshape(net, shape)
        bbox_pred = net[:,:,:,:,:4]
        obj_score = net[:,:,:,:,4]
        obj_prob = tf.sigmoid(obj_score)
        cls_score = net[:,:,:,:,5:]
        cls_prob = tf.nn.softmax(cls_score)
    
        yolo_layers.append({'bbox_pred':bbox_pred,
                            'obj_score':obj_score, 'obj_prob':obj_prob,
                            'cls_score':cls_score, 'cls_prob':cls_prob})
    return yolo_layers

# labels:         [batch,row,col,box],   int32
# obj_cond:       [batch,row,col,box],   bool
# noobj_cond:     [batch,row,col,box],   bool
# [no]obj_bbox_t: [batch,row,col,box,4], float32
def yolo_loss(end_points, labels, obj_cond, noobj_cond,
              obj_bbox_t, noobj_bbox_t=[[[[[.5,.5,0.,0.]]]]]):
    obj_cond = tf.cast(obj_cond, tf.float32)
    noobj_cond = tf.cast(noobj_cond, tf.float32)
    
    obj_prob = end_points['obj_prob']
    obj_loss = obj_cond * (obj_prob - 1.)**2/2
    noobj_loss = noobj_cond * (obj_prob - 0.)**2/2

    bbox_pred = end_points['bbox_pred']
    xy_pred = tf.sigmoid(bbox_pred[:,:,:,:,:2])
    wh_pred = bbox_pred[:,:,:,:,2:]
    bbox_pred = tf.concat([xy_pred, wh_pred], -1)
    bbox_loss = obj_cond * tf.reduce_sum((bbox_pred - obj_bbox_t)**2, -1)/2
    if noobj_bbox_t is not None:
        bbox_loss += noobj_cond * .01 \
                     * tf.reduce_sum((bbox_pred - noobj_bbox_t)**2, -1)/2

    cls_score = end_points['cls_score']
    cls_loss = obj_cond * tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, labels=labels)

    return {'bbox_loss': bbox_loss,
            'obj_loss': obj_loss, 'noobj_loss': noobj_loss,
            'cls_loss': cls_loss}
