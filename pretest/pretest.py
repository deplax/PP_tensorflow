# -*- coding: utf-8 -*-
__author__ = 'whale'
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import tensorflow as tf

hello = tf.constant("hello, tensorflow!")
sess = tf.Session()
print sess.run(hello)

import tensorflow as tf
sess = tf.Session()
w = tf.Variable(tf.random_normal([3, 3]), name = "w")
y = tf.matmul(x, w)
relu_out = tf.nn.relu(y)
print sess.run(relu_out)