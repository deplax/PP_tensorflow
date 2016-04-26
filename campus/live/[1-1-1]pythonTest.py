# -*- coding: utf-8 -*-
__author__ = 'whale'
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import os
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

IMG_DIR = "img"
PROJECT_DIR = "/Users/whale/private/tensorflow/"
IMG_ABS_DIR = os.path.join(PROJECT_DIR, IMG_DIR)


imgs = []
names = []
path = os.path.join(IMG_ABS_DIR, "cat")
FILENAME = "/cat.jpg"
valid_exts = [".jpg", ".gif", ".png", "tga"]

print os.listdir(path)


for f in os.listdir(path):
    print "f: ", f
    ext = os.path.splitext(f)[1]
    print "ext: ", ext
    if ext.lower() not in valid_exts:
        continue
    fullpath = os.path.join(path, f)
    imgs.append(misc.imread(fullpath))
    names.append(os.path.splitext(f)[0] + os.path.split(f)[1])


print "Type of 'imgs' : ", type(imgs)
print "Length of 'imgs' : ", len(imgs)

i = 0
for curr_img in imgs:
    i = i + 1
    print ""
    print i, "Type of 'curr_img': ", type(curr_img)
    print i, "Size of 'curr_img': %s " % (curr_img.shape, )
