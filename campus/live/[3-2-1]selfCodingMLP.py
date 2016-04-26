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

# 이미지를 불러오지요
mnist = input_data.read_data_sets("data/", one_hot=True)

# 불러왔는지 보자
print mnist

trainimgs = mnist.train.images
trainlabel = mnist.train.labels
testimgs = mnist.test.images
testlabel = mnist.test.labels

ntrain = trainimgs.shape[0]
print ntrain
ntest = testimgs.shape[0]
print ntest
dim = trainimgs.shape[1]
print dim  # 28 * 28
nout = trainlabel.shape[1]
print nout

print "number of train imgs is %d, dim is %d" % (ntrain, dim)


# 우린 멀티 레이어 퍼셉트론을 만들겁니다.
nhid1 = 256
nhid2 = 128


# 이제 웨이트 구조를 만들겁니다.

# 784 -> 256 -> 128 -> 10
# 이렇게 되어 weight가 3개 필요하다 (화살표 자리)
# bias도 역시 3개가 필요하다.

weights = {
    # 가우시안 분포에서 뽑는다.
    "h1": tf.Variable(tf.random_normal([dim, nhid1], stddev=0.1)),
    "h2": tf.Variable(tf.random_normal([nhid1, nhid2], stddev=0.1)),
    "out": tf.Variable(tf.random_normal([nhid2, nout], stddev=0.1))
}

biases = {
    "b1": tf.Variable(tf.random_normal([nhid1], stddev=0.1)),
    "b2": tf.Variable(tf.random_normal([nhid2], stddev=0.1)),
    "out": tf.Variable(tf.random_normal([nout], stddev=0.1))
}


# 이제 네트워크를 만들자요
# multiLayerPerceptron
def mlp(_X, _W, _b):
    _layer1 = tf.nn.relu(tf.add(tf.matmul(_X, _W["h1"]), _b["b1"]))
    _layer2 = tf.nn.relu(tf.add(tf.matmul(_layer1, _W["h2"]), _b["b2"]))
    _out = tf.add(tf.matmul(_layer2, _W["out"]), _b["out"])
    return _out


# learning parameter
learning_rate = 0.001
training_epochs = 10
batch_size = 100
display_step = 1

# define functions

x = tf.placeholder(tf.float32, [None, dim])
y = tf.placeholder(tf.float32, [None, nout])
pred = mlp(x, weights, biases)

# prediction이 나왔을 때
# prediction과 y값을 비교할 수 있는게 필요 그래서 cost function을 만듬

# 데이터가 한번에 하나가 들어가는게 아니라서 tf.reduce_mean추가
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))

# cost function이 생겼으니 optimazer를 만들자.
optm = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# 두 값의 같은 개수를 저장.
corr = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# cast는 형을 맞추려고
accr = tf.reduce_mean(tf.cast(corr, tf.float32))

init = tf.initialize_all_variables()

# 이제 다 만들었다. 출력을 만들자.
print "Network Ready ========================================="

sess = tf.Session()
sess.run(init)

for epoch in range(training_epochs):
    total_cost = 0;
    total_batch = int(ntrain / batch_size)
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(optm, feed_dict={x: batch_xs, y: batch_ys})
        curr_cost = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})
        total_cost = total_cost + curr_cost
    avg_cost = total_cost / total_batch

    # Display
    if epoch % display_step == 0:
        print "Epoch: %d, cost: %.3f" % (epoch, avg_cost)
        train_acc = sess.run(accr, feed_dict={x: batch_xs, y: batch_ys})
        test_acc = sess.run(accr, feed_dict={x: batch_xs, y: batch_ys})
        print "Training accr: %.3f Test accr: %.3f" % (train_acc, test_acc)


print "Optinization Finished"