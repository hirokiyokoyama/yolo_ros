#####################################
#resize image into 10 size and save #
#####################################

import cv2
import numpy as np
import sys, os

image_path = sys.argv[1]

imgs = os.listdir(image_path)

def resize_img(img, name):
    try:
        image = cv2.imread(image_path + img, cv2.IMREAD_UNCHANGED)
    except:
        if image is None:
            print "Failed to load image file."
        else:
            print "This file is not image. Can not open this file."
    orgH, orgW = image.shape[:2]
    size_param = (3 - 0.3) * np.random.rand(10) + 0.3
    
    num = 0
    for i in size_param:
        #size = (int(orgH*i), int(orgW*i))
        resize_img = cv2.resize(image, None, fx=i, fy=i, interpolation = cv2.INTER_CUBIC)
        cv2.imwrite(image_path + name + "-" + str(num+1) + ".png", resize_img)
        num += 1
def main(imgs):
    for n in range(len(imgs)):
        image = imgs[n]
        name = image.strip(".png")
        resize_img(image, name)

main(imgs)
