# -*- coding: utf-8 -*-
__author__ = 'whale'
import sys

reload(sys)
sys.setdefaultencoding('utf-8')






"""
 Logistic Regression with MNIST
 @Sungjoon Choi (sungjoon.choi@cpslab.snu.ac.kr)
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data








"""
 Download and Extract MNIST dataset
"""
mnist      = input_data.read_data_sets('../data/', one_hot=True)
trainimg   = mnist.train.images
trainlabel = mnist.train.labels
testimg    = mnist.test.images
testlabel  = mnist.test.labels









"""
 Parameters of Logistic Regression
"""
learning_rate   = 0.01
training_epochs = 20
# 5만 5천개를
# 100개씩 떼서 넣습니다.
batch_size      = 100
display_step    = 1

"""
 Create Graph for Logistic Regression
"""
x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float", [None, 10])  # None is for infinite or unspecified length
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

"""
 Activation, Cost, and Optimizing functions
"""

#activation function인데 어느 클래스에 속할 확률이다. 를 만들어줌.
#softmax는 어떤 값이 들어오면 0과 1 사이로 만들어준다.
activation = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax


## 목표치와 내가 계산한 값의 차이의 제곱을 최소로 맞춰줌 (아까꺼)



# 크로스 엔트로피를 의미
# 두 확률분포 사이의 거리를 정의하는 것이 크로스 엔트로피
#
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(activation), reduction_indices=1)) # Cross entropy
# cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(activation) + (1-y)*tf.log(1-activation), reduction_indices=1)) # Cross entropy
# cost = tf.reduce_mean(tf.pow(activation-y, 2)) 이건 뭐 좀 볼라고 넣은거

# * is an element-wise product in numpy (in Matlab, it should be .*)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) # Gradient Descent

"""
 Optimize with TensorFlow
"""
# Initializing the variables
init = tf.initialize_all_variables()












# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        num_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(num_batch):
            if 0: # Using tensorflow API
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            else: # Random batch sampling
                randidx = np.random.randint(trainimg.shape[0], size=batch_size)
                batch_xs = trainimg[randidx, :]
                batch_ys = trainlabel[randidx, :]

                # Fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            # Compute average loss
            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})/num_batch


        # Display logs per epoch step
        if epoch % display_step == 0:
            print ("Epoch: %03d/%03d cost: %.9f" % (epoch, training_epochs, avg_cost))

    print ("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(activation, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print ("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

print ("Done.")