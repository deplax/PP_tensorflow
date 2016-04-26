# -*- coding: utf-8 -*-
__author__ = 'whale'
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# 해당 위치에 mnist 샘플데이터셋을 받는다.
# one_hot 은 라벨을 0과 1로만 붙이는 설정이다.
# 해당 디렉토리는 생성되어 있지 않아도 자동으로 생성하며, 해당 파일이 존재할 경우 다운로드 과정이 제외된다.
mnist = input_data.read_data_sets('mnist/data/', one_hot=True)

# 트레이닝용과 테스트용을 각각 준비한다.
trainimg = mnist.train.images
trainlabel = mnist.train.labels
testimg = mnist.test.images
testlabel = mnist.test.labels

# (55000, 784)
print trainimg.shape


# lower cost를 찾기 위해 이동하는 스텝.
learning_rate = 0.01
# 전체 몇 번을 수행할지
training_epochs = 500
# learning을 하기위해 한번에 들어가는 양.
batch_size = 100
# 몇번마다 출력할지
display_step = 1


# 레이어의 아웃풋을 256으로 잡았다.
n_hidden_1 = 256
n_hidden_2 = 256
# mnist의 이미지 한장은 28 * 28로 되어 있다.
n_input = 784
# 최종적으로 다시 같은 이미지를 얻을 것이니 인펏과 같다.
n_output = 784

# input의 dimension으로 placeholder를 만든다.
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_output])
dropout_keep_prob = tf.placeholder("float")


def denoising_autoencoder(_X, _weights, _biases, _keep_prob):

    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1']))
    layer_1_out = tf.nn.dropout(layer_1, _keep_prob)


print dropout_keep_prob