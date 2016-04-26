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

mnist = input_data.read_data_sets("data/", one_hot=True)

trainimgs = mnist.train.images
trainlabel = mnist.train.labels
testimgs = mnist.test.images
testlabel = mnist.test.labels

ntrain = trainimgs.shape[0]
ntest = testimgs.shape[0]
dim = trainimgs.shape[1]
nout = trainlabel.shape[1]

# 이제 cnn을 짤겁니다.
# convelution이니 이미지가 그대로 살아있다.

# 우린 3바이 3짜리 필터 64개를 쓸꺼다.
# 2바이 2 맥스폴링을 할꺼다 겹치는거 없이 2개씩 뛴다.
# 그리고 한줄로 펴고 (reshape)
# 128 히든레이어로 만들고
# 10으로 만들꺼다.

# 위쪽은 MLP랑 같다.

# conv 하고 bias 더하고 activation을 통과 시킨다.
# bias는 필터 갯수만큼임.


conv1_h = 3
conv1_w = 3
conv1_n = 64

nhid1 = 128
weights = {
    # 세로, 가로, 채널(mnist는 흑백), 아웃풋 채널
    "wc1": tf.Variable(tf.random_normal([conv1_h, conv1_w, 1, conv1_n], stddev=0.1)),
    "wd1": tf.Variable(tf.random_normal([14 * 14 * 64, nhid1], stddev=0.1)),  # 덴스레이어
    "out": tf.Variable(tf.random_normal([nhid1, nout], stddev=0.1)),
}

biases = {
    "bc1": tf.Variable(tf.random_normal([conv1_n], stddev=0.1)),
    "bd1": tf.Variable(tf.random_normal([nhid1], stddev=0.1)),
    "out": tf.Variable(tf.random_normal([nout], stddev=0.1))
}


def cnn(_X, _W, _b):
    _input_r = tf.reshape(_X, shape=[-1, 28, 28, 1])
    # Conv1
    # 컨벌루션 필터에 맞춰서 패딩을 줘라
    _conv1a = tf.nn.conv2d(_input_r, _W["wc1"], strides=[1, 1, 1, 1], padding="SAME")
    _conv1b = tf.nn.bias_add(_conv1a, _b["bc1"])
    _conv1c = tf.nn.relu(_conv1b)

    # Pool 1
    _pool1 = tf.nn.max_pool(_conv1c, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # Vectorize
    _dense = tf.reshape(_pool1, [-1, _W["wd1"].get_shape().as_list()[0]])

    # FC1
    _fc = tf.nn.relu(tf.add(tf.matmul(_dense, _W["wd1"]), _b["bd1"]))

    _out = tf.add(tf.matmul(_fc, _W["out"]), _b["out"])

    return _out


# 이제 네트워크는 다 만들었어요(가장 간단한 CNN)
# 이 뒤는 완전히 똑같아.

# Define functinos
learning_rate = 0.001
training_epochs = 10
batch_size = 100
display_step = 1

x = tf.placeholder(tf.float32, [None, dim])
y = tf.placeholder(tf.float32, [None, nout])

pred = cnn(x, weights, biases)
cost = tf.nn.softmax_cross_entropy_with_logits(pred, y)
optm = tf.train.AdamOptimizer(learning_rate).minimize(cost)

corr = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accr = tf.reduce_mean(tf.cast(corr, tf.float32))
init = tf.initialize_all_variables()

print "Network Ready"

sess = tf.Session()
sess.run(init)

for epoch in range(training_epochs):
    total_cost = 0
    total_batch = int(ntrain / batch_size)
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(optm, feed_dict={x: batch_xs, y: batch_ys})
        curr_cost = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})
        total_cost = total_cost + curr_cost
    avg_cost = total_cost / total_batch

    # Display
    if epoch % display_step == 0:
        print "Epoch: %d" % (epoch)
        train_acc = sess.run(accr, feed_dict={x: batch_xs, y: batch_ys})
        test_acc = sess.run(accr, feed_dict={x: batch_xs, y: batch_ys})
        print "Training accr: %.3f Test accr: %.3f" % (train_acc, test_acc)


print "Optinization Finished"