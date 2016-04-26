__author__ = 'whale'

"""
 Logistic Regression with Custom Data
 @Sungjoon Choi (sungjoon.choi@cpslab.snu.ac.kr)
"""
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt



# Load them!
cwd = os.getcwd()
loadpath = cwd + "/data/trainingset.npz"
l = np.load(loadpath)

# See what's in here
l.files

# Parse data
trainimg = l['trainimg']
trainlabel = l['trainlabel']
testimg = l['testimg']
testlabel = l['testlabel']
ntrain = trainimg.shape[0]
nclass = trainlabel.shape[1]
dim    = trainimg.shape[1]
ntest  = testimg.shape[0]

print ("%d train images loaded" % (ntrain))
print ("%d test images loaded" % (ntest))
print ("%d dimensional input" % (dim))
print ("%d classes" % (nclass))




# Parameters of Logistic Regression
learning_rate   = 0.001
training_epochs = 1000
batch_size      = 10
display_step    = 10

# Create Graph for Logistic Regression
x = tf.placeholder("float", [None, dim])
y = tf.placeholder("float", [None, nclass])  # None is for infinite or unspecified length
W = tf.Variable(tf.zeros([dim, nclass]))
b = tf.Variable(tf.zeros([nclass]))

# Activation, Cost, and Optimizing functions
_pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(_pred), reduction_indices=1)) # Cross entropy
# cost = tf.reduce_mean(tf.pow(activation-y, 2))
optm = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) # Gradient Descent
_corr = tf.equal(tf.argmax(_pred, 1), tf.argmax(y, 1))
accr = tf.reduce_mean(tf.cast(_corr, tf.float32))
init = tf.initialize_all_variables()

print ("Network Ready")



# Launch the graph
sess = tf.Session()
sess.run(init)

# Training cycle
for epoch in range(training_epochs):
    avg_cost = 0.
    num_batch = int(ntrain/batch_size)
    # Loop over all batches
    for i in range(num_batch):
        randidx = np.random.randint(ntrain, size=batch_size)
        batch_xs = trainimg[randidx, :]
        batch_ys = trainlabel[randidx, :]
        # Fit training using batch data
        sess.run(optm, feed_dict={x: batch_xs, y: batch_ys})
        # Compute average loss
        avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})/num_batch

    # Display logs per epoch step
    if epoch % display_step == 0:
        print ("Epoch: %03d/%03d cost: %.9f" % (epoch, training_epochs, avg_cost))
        train_acc = sess.run(accr, feed_dict={x: batch_xs, y: batch_ys})
        print (" Training accuracy: %.3f" % (train_acc))
        test_acc = sess.run(accr, feed_dict={x: testimg, y: testlabel})
        print (" Test accuracy: %.3f" % (test_acc))

print ("Optimization Finished!")