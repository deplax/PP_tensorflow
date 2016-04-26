# -*- coding: utf-8 -*-
__author__ = 'whale'
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import tensorflow as tf
import numpy as np

# 2 x 100 의 메트락스 생성. 각각 0~1 사이의 값이 들어간다.
x_data = np.float32(np.random.rand(2, 2))



# 0.1, 0.2를 각각 2 x 100 에 곱하고 + 0.3
y_data = np.dot([0.100, 0.200], x_data) + 0.300
# 1 x 100 의 벡터가 나온다

print x_data, "\n"
print y_data

# 바이어스
b = tf.Variable(tf.zeros([1]))

# 가중치 행렬
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))

y = tf.matmul(W, x_data) + b

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

##
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for step in xrange(0, 201):
    sess.run(train)
    if step % 20 == 0:
        print step, sess.run(W), sess.run(b)
