#!/usr/bin/env python

import rospy
import cv2
import numpy as np
import cv_bridge
from sensor_msgs.msg import Image

class Yolo9000Train:
    def __init__(self, bg_dir, fg_dir):
        self._find_files(bg_dir, fg_dir)
        self.num_boxes = 3
        self.anchor_ws = np.array([0.77871, 3.00525, 9.22725])
        self.anchor_hs = np.array([1.14074, 4.31277, 9.61974])

        self.obj_scale = 5
        self.noobj_scale = 1
        self.class_scale = 1
        self.coord_scale = 1
        self.learning_rate = 0.0001
        self.hue = .1
        self.saturation = .75
        self.exposure = .75

    def _find_files(self, bg_dir, fg_dir):
        import os
        isimg = lambda f: f.endswith('.png') or f.endswith('.jpg')

        self.bg_files = [os.path.join(bg_dir, f)
                         for f in os.listdir(bg_dir) if isimg(f)]
        self.fg_files = {}
        self.class_names = []
        for f in os.listdir(fg_dir):
            if not isimg(f):
                continue
            name = '.'.join(f.split('.')[:-1])
            name = '_'.join(name.split('_')[:-1])
            if name not in self.class_names:
                self.class_names.append(name)
                self.fg_files[name] = []
            self.fg_files[name].append(os.path.join(fg_dir, f))
            
        self.num_classes = len(self.class_names)
        print self.class_names
        print self.fg_files

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

    def create_image_batch(self, size=1, num_objs=3):
        scale = np.random.randint(10, 15)
        imsize = 32*scale

        img = np.zeros((size, imsize, imsize, 3), dtype=np.float32)
        bbox = np.zeros((size, num_objs, 4), dtype=np.float32)
        cls = np.zeros((size, num_objs), dtype=np.int32)

        for i in xrange(size):
            bg_file = np.random.choice(self.bg_files)
            bg_img = self.distort_image(cv2.imread(bg_file))/255.

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

            for j, cind in enumerate(np.random.choice(self.num_classes, num_objs)):
                cname = self.class_names[cind]
                fg_file = np.random.choice(self.fg_files[cname])
                fg_img = self.distort_image(cv2.imread(fg_file, -1))/255.
                if fg_img.shape[2]==4:
                    fg_alpha = fg_img[:,:,3:]
                    fg_img = fg_img[:,:,:3]
                else:
                    fg_alpha = 1.
                fg_img = fg_img[:,:,::-1] #BGR -> RGB

                dw = imsize - fg_img.shape[1]
                dh = imsize - fg_img.shape[0]
                if dw <= 0 or dh <= 0:
                    print 'Object image is too large.'
                    return None
                w = fg_img.shape[1]
                h = fg_img.shape[0]
                left = np.random.randint(0, dw)
                right = left + w
                top = np.random.randint(0, dh)
                bottom = top + h
                img[i,top:bottom,left:right,:] \
                    = img[i,top:bottom,left:right,:]*(1-fg_alpha) + fg_img*fg_alpha
                
                x = (left+right)/2./imsize
                y = (top+bottom)/2./imsize
                w = float(w)/imsize
                h = float(h)/imsize
                bbox[i,j,:] = [x,y,w,h]
                cls[i,j] = cind
        return img, bbox, cls

import sys
print 'Background image dir : ' + sys.argv[1]
print 'Object image dir     : ' + sys.argv[2]

rospy.init_node('yolo9000_training_image')
pub = rospy.Publisher('image', Image, queue_size=10)
train = Yolo9000Train(sys.argv[1], sys.argv[2])
bridge = cv_bridge.CvBridge()
rate = rospy.Rate(.5)
while not rospy.is_shutdown():
    img, bbox, cls = train.create_image_batch(size=1, num_objs=3)
    pub.publish(bridge.cv2_to_imgmsg(np.uint8(img[0,:,:,::-1]*255),'bgr8'))
    rate.sleep()
