# -*- coding: utf-8 -*-
__author__ = 'whale'
import sys

reload(sys)
sys.setdefaultencoding('utf-8')

import numpy as np
import os
from scipy import misc
import matplotlib.pyplot as plt


def rgb2gray(rgb):
    return np.dot(rgb[..., : 3], [0.299, 0.587, 0.144])


IMG_DIR = "img"
PROJECT_DIR = "/Users/whale/private/tensorflow/"
IMG_ABS_DIR = os.path.join(PROJECT_DIR, IMG_DIR)

FILENAME = "/cat.jpg"

cat = misc.imread(IMG_ABS_DIR + FILENAME)
# cat = cat[:, :, 0]
# cat = rgb2gray(cat)

cat = misc.imresize(cat, [100, 100, 1])

catrowvec = np.reshape(cat, (1, -1))

catrowvec2 = np.reshape(cat, (100, -1))

print "Type of cat is %s" % (type(cat))
print "Shape of cat is %s" % (cat.shape,)
print "Shape of cat is %s" % (catrowvec.shape,)
print "Shape of cat is %s" % (catrowvec2.shape,)

# plt.figure(1)
# plt.imshow(cat, cmap=plt.get_cmap("gray"))
# plt.title("Original Image")

# a = plt.matshow(cat, fignum=1, cmap=plt.get_cmap("gray"))
# a = plt.matshow(cat, fignum=1, cmap=plt.get_cmap("jet"))
# plt.colorbar(a)



plt.figure(1)
plt.imshow(cat)
plt.show()
