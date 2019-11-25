import tensorflow as tf
layers = tf.keras.layers

DEFAULT_ANCHORS = tf.constant([[50, 50]])

class YoloOutputLayer(layers.Layer):
  def __init__(self, num_boxes):
    super(YoloOutputLayer, self).__init__()
    self._num_boxes = num_boxes

  def call(self, x):
    n, h, w, c = tf.unstack(tf.shape(x))
    c //= self._num_boxes
    return tf.reshape(x, [n, h, w, self._num_boxes, c])

class Conv2D(layers.Conv2D):
  def __init__(self, *args, **kwargs):
    kwargs['kernel_initializer'] = tf.keras.initializers.RandomNormal(stddev=1.)
    super(Conv2D, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    super(Conv2D, self).build(input_shape)
    shape = self.kernel.shape.as_list()
    self._normalization_constant = np.sqrt(2./(shape[0] * shape[1] * shape[2]))
  
  def call(self, x):
    x = x * self._normalization_constant
    x = super(Conv2D, self).call(x)
    return x

class YoloOutputCodec(object):
  def __init__(self, image_size, anchors=DEFAULT_ANCHORS):
    self._anchors = tf.reshape(anchors, [1,1,1,-1,2])
    self._img_w, self._img_h = image_size

  def decode(self, yolo_output):
    data = yolo_output

    # decode w and h
    w, h = tf.unstack(tf.exp(data[:,:,:,:,2:]) * self._anchors, 4)

    # decode x and y
    _, grid_h, grid_w, _ = tf.unstack(tf.shape(data))
    cx, cy = tf.unstack(tf.nn.sigmoid(data[:,:,:,:,:2]), 4)
    dx, dy = tf.meshgrid(tf.range(grid_w, dtype=tf.float32), tf.range(grid_h, dtype=tf.float32))
    cx = (cx + tf.reshape(dx, [1, grid_h, grid_w, 1])) * self._img_w / grid_w
    cy = (cy + tf.reshape(dy, [1, grid_h, grid_w, 1])) * self._img_h / grid_h

    x = cx - w/2
    y = cy - h/2

    # decode objectness
    obj = tf.nn.sigmoid(data[:,:,:,:,4])

    # decode class probabilities
    cls = tf.nn.softmax(data[:,:,:,:,5:])

    return x, y, w, h, obj, cls

  def encode(self, x, y, w, h, obj, cls):
    # encode x and y
    cx = x + w/2
    cy = y + h/2
    dx, dy = tf.meshgrid(tf.range(grid_w, dtype=tf.float32), tf.range(grid_h, dtype=tf.float32))
    cx = cx * grid_w / self._img_w - tf.reshape(dx, [1, grid_h, grid_w, 1])
    cy = cy * grid_h / self._img_h - tf.reshape(dy, [1, grid_h, grid_w, 1])

    # encode w and h
    w, h = tf.unstack(tf.log(tf.stack([w, h], 4) / self._anchors), 4)

    # leave obj and cls as they are (used as labels for cross-entropy)

    return tf.concat([tf.stack([cx, cy, w, h, obj], 4), cls], 4)

class YoloLoss(tf.keras.losses.Loss):
  def __init__(self,
      obj_scale = 4., noobj_scale = 1., bbox_scale = 1., cls_scale = 1.,
      noobj_bbox=[0.5, 0.5, 0., 0.],
      name=None):
    super(YoloLoss, self).__init__(name=name)
    self._obj_scale = obj_scale
    self._noobj_scale = noobj_scale
    self._bbox_scale = bbox_scale
    self._cls_scale = cls_scale
    self._noobj_bbox = noobj_bbox

  def call(self, y_true, y_pred):
    bbox_true = y_true[:,:,:,:,:4]
    bbox_pred = y_pred[:,:,:,:,:4]
    obj_true = y_true[:,:,:,:,4]
    obj_pred = y_pred[:,:,:,:,4]
    cls_true = y_true[:,:,:,:,5:]
    cls_pred = y_pred[:,:,:,:,5:]

    #obj_mask = tf.cast(tf.greater_equal(obj_true, 0.), tf.float32)
    obj_scale = obj_true * self._obj_scale + (1.-obj_true) * self._noobj_scale
    obj_loss = obj_scale * (obj_true - obj_pred)**2/2

    xy_pred = tf.nn.sigmoid(bbox_pred[:,:,:,:,:2])
    wh_pred = bbox_pred[:,:,:,:,2:]
    bbox_pred = tf.concat([xy_pred, wh_pred], 4)
    bbox_loss = obj_true * tf.reduce_sum((bbox_pred - bbox_true)**2, 4)/2
    if self._noobj_bbox is not None:
        bbox_loss += (1.-obj_true) * .01 \
                     * tf.reduce_sum((bbox_pred - self._noobj_bbox)**2, 4)/2
    bbox_loss = self._bbox_scale * bbox_loss

    cls_loss = obj_true * tf.nn.softmax_cross_entropy_with_logits(logits=cls_pred, labels=cls_true)
    cls_loss = self._cls_scale * cls_loss

    return tf.add_n([obj_loss, bbox_loss, cls_loss])

class Yolo(tf.keras.Sequential):
    def __init__(self, num_classes, num_boxes=1):
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        
        c = num_boxes * (4 + 1 + num_classes)

        super().__init__([
            Conv2D(32, 3, 1, 'SAME'),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.MaxPooling2D(2),

            Conv2D(64, 3, 1, 'SAME'),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.MaxPooling2D(2),

            Conv2D(128, 3, 1, 'SAME'),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            Conv2D(64, 1, 1, 'SAME'),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            Conv2D(128, 3, 1, 'SAME'),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.MaxPooling2D(2),

            Conv2D(256, 3, 1, 'SAME'),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            Conv2D(128, 1, 1, 'SAME'),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            Conv2D(256, 3, 1, 'SAME'),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.MaxPooling2D(2),

            Conv2D(512, 3, 1, 'SAME'),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            Conv2D(256, 1, 1, 'SAME'),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            Conv2D(512, 3, 1, 'SAME'),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.MaxPooling2D(2),

            Conv2D(1024, 3, 1, 'SAME'),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            Conv2D(512, 1, 1, 'SAME'),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            Conv2D(1024, 3, 1, 'SAME'),
            layers.BatchNormalization(),
            layers.LeakyReLU(),

            Conv2D(1024, 3, 1, 'SAME'),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            Conv2D(c, 1, 1, 'SAME'),
            YoloOutputLayer(num_boxes)
        ])
