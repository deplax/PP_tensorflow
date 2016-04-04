# -*- coding: utf-8 -*-
__author__ = 'whale'
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

__author__ = 'whale'

import tensorflow as tf

hello = tf.constant("hello, tensorflow!")
sess = tf.Session()
print sess.run(hello)