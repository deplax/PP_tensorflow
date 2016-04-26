# -*- coding: utf-8 -*-
__author__ = 'whale'
import sys

reload(sys)
sys.setdefaultencoding('utf-8')

# ==================================================================

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('data/', one_hot=True)
trainimg = mnist.train.images
trainlabel = mnist.train.labels
testimg = mnist.test.images
testlabel = mnist.test.labels

# 784 layer3 x
# w3
# 256 layer2 keepprob
# w2
# 256 layer1 keepprob
# w1
# 784 x

weights = {
    "w1": tf.Variable(tf.random_normal([784, 256], stddev=0.01)),
    "w2": tf.Variable(tf.random_normal([256, 256], stddev=0.01)),
    "w3": tf.Variable(tf.random_normal([256, 784], stddev=0.01))
}

biases = {
    "b1": tf.Variable(tf.random_normal([256], stddev=0.01)),
    "b2": tf.Variable(tf.random_normal([256], stddev=0.01)),
    "b3": tf.Variable(tf.random_normal([784], stddev=0.01))
}


def _dae(_X, _W, _b, _keepprob, noiseFlag):
    if noiseFlag is True:
        _X = tf.add(_X + tf.random_normal([784], stddev=0.1))
    _layer1a = tf.nn.relu(tf.add(tf.matmul(_X, _W["w1"]), _b["b1"]))
    _layer1b = tf.nn.dropout(_layer1a, _keepprob)
    _layer2a = tf.nn.relu(tf.add(tf.matmul(_layer1b, _W["w2"]), _b["b2"]))
    _layer2b = tf.nn.dropout(_layer2a, _keepprob)
    _layer3a = tf.nn.sigmoid(tf.add(tf.matmul(_layer2b, _W["w3"]), _b["b3"]))

    outbox = {
        "l1": _layer1b,
        "l2": _layer2b,
        "out": _layer3a,
    }

    return outbox


# placeholders

x = tf.placeholder(tf.float32, [None, 784])
# y는 필요없다. 왜냐면 오토인코더니까

# keepprob을 만드는 이유는 테스트시에는 드랍아웃을 하면 안되니까.
keepprob = tf.placeholder(tf.float32)
noiseFlag = tf.placeholder(tf.int32)

out = _dae(x, weights, biases, keepprob, noiseFlag)["out"]
# 제곱의 평균을 낸다
cost = tf.reduce_mean(tf.pow(out - x, 2))
optm = tf.train.AdamOptimizer(0.001).minimize(cost)
init = tf.initialize_all_variables()

print ("network ready")

# ==========================================================================

learning_rate = 0.001
training_epochs = 5
batch_size = 100
display_step = 20

sess = tf.Session()
sess.run(init)

for epoche in range(training_epochs):
    total_cost = 0
    # 몇번 돌릴지
    num_batch = int(trainimg.shape[0] / batch_size)
    sum_cost = 0
    for i in range(num_batch):
        batch_xs, _ = mnist.train.next_batch(batch_size)
        sess.run(optm, feed_dict={x: batch_xs, keepprob: 0.7, noiseFlag: True})
        sum_cost = sum_cost + sess.run(cost, feed_dict={x: batch_xs, keepprob: 1., noiseFlag: False})
    avg_cost = sum_cost / num_batch
    print ("epoch: %d avg_cost: %.4f" % (epoche, avg_cost))



# 다른레이어 빼오기
batch_xs, _ = mnist.train.next_batch(batch_size)
l2val = sess.run(_dae(x, weights, biases, keepprob)["l2"], feed_dict={x: batch_xs, keepprob: 0.7})
