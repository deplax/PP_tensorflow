# -*- coding: utf-8 -*-
__author__ = 'whale'
import sys

reload(sys)
sys.setdefaultencoding('utf-8')

# ==================================================================

# cnn을 좀 볼 수 있는 구조로 만들어보자.

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("data/", one_hot=True)
trainimg = mnist.train.images
trainlabel = mnist.train.labels
testimg = mnist.test.images
testlabel = mnist.test.labels


# Parameters
learning_rate = 0.01
training_epochs = 50
batch_size = 100
display_step = 1
plot_step = 10

use_dropout = 1

# Network Parameters
n_hidden_1 = 256  # 1st layer num features
n_hidden_2 = 256  # 2nd layer num features
n_input = 784  # MNIST data input (img shape: 28*28)
n_output = 784  #

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_output])
dropout_keep_prob = tf.placeholder("float")


# Create model
def denoising_autoencoder(_X, _weights, _biases, _keep_prob):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1']))
    layer_1out = tf.nn.dropout(layer_1, _keep_prob)
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1out, _weights['h2']), _biases['b2']))
    layer_2out = tf.nn.dropout(layer_2, _keep_prob)
    return tf.nn.sigmoid(tf.matmul(layer_2out, _weights['out']) + _biases['out'])


# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_output]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_output]))
}

# Construct model
out = denoising_autoencoder(x, weights, biases, dropout_keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.pow(out - y, 2))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)  # Adam Optimizer
# optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(cost) # Momentum

# Initializing the variables
init = tf.initialize_all_variables()

# Saver
savedir = "nets/"
saver = tf.train.Saver(max_to_keep=training_epochs)

# Launch the graph
sess = tf.Session()

print ("Network Ready")

"""
 Don't run this cell unless you want to train all over
"""
do_train = 0

# Training
sess.run(init)

if do_train:

    print ("Start Optimization")
    for epoch in range(training_epochs):
        avg_cost = 0.
        num_batch = int(mnist.train.num_examples / batch_size)
        # Loop over all batches
        for i in range(num_batch):
            randidx = np.random.randint(trainimg.shape[0], size=batch_size)
            batch_xs = trainimg[randidx, :]
            batch_xs_noisy = batch_xs + 0.3 * np.random.randn(batch_xs.shape[0], 784)

            batch_ys = trainlabel[randidx, :]

            # Fit training using batch data
            if use_dropout:
                sess.run(optimizer, feed_dict={x: batch_xs_noisy, y: batch_xs, dropout_keep_prob: 0.5})
            else:
                sess.run(optimizer, feed_dict={x: batch_xs_noisy, y: batch_xs, dropout_keep_prob: 1.})
            # Compute average loss
            avg_cost += sess.run(cost, feed_dict={x: batch_xs_noisy, y: batch_xs, dropout_keep_prob: 1}) / num_batch

        # Display logs per epoch step
        if epoch % display_step == 0:
            print ("Epoch: %03d/%03d cost: %.9f" % (epoch, training_epochs, avg_cost))
        if epoch % plot_step == 0:
            # Test one
            randidx = np.random.randint(testimg.shape[0], size=1)
            testvec = testimg[randidx, :]
            noisyvec = testvec + 0.3 * np.random.randn(1, 784)
            outvec = sess.run(out, feed_dict={x: testvec, dropout_keep_prob: 1.})
            outimg = np.reshape(outvec, (28, 28))

            # Plot
            plt.matshow(np.reshape(testvec, (28, 28)), cmap=plt.get_cmap('gray'))
            plt.title("[" + str(epoch) + "] Original Image")
            plt.colorbar()

            plt.matshow(np.reshape(noisyvec, (28, 28)), cmap=plt.get_cmap('gray'))
            plt.title("[" + str(epoch) + "] Input Image")
            plt.colorbar()

            plt.matshow(outimg, cmap=plt.get_cmap('gray'))
            plt.title("[" + str(epoch) + "] Reconstructed Image")
            plt.colorbar()
            plt.show()

            # Save
            if use_dropout:
                saver.save(sess, savedir + 'dae_dr.ckpt', global_step=epoch)
            else:
                saver.save(sess, savedir + 'dae.ckpt', global_step=epoch)
    print ("Optimization Finished!")
