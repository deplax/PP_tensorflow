# -*- coding: utf-8 -*-
__author__ = 'whale'
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

"""
 Let's get friendly with MNIST
 @Sungjoon Choi (sungjoon.choi@cpslab.snu.ac.kr)
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data







"""
 Download and Extract MNIST dataset
"""
print ("1. Download and Extract MNIST dataset")
# ont_hot coding
# 내가 원하는 출력값이 여러개의 클레스에서 하나만 1인 것.
mnist = input_data.read_data_sets('../data/', one_hot=True)
print (" tpye of 'mnist' is ", type(mnist))
print (" number of trian data is %d" % (mnist.train.num_examples))
print (" number of test data is %d" % (mnist.test.num_examples))







"""
 What does the data of MNIST look like?
"""
print ("2. What does the data of MNIST look like?")
trainimg   = mnist.train.images
trainlabel = mnist.train.labels
testimg    = mnist.test.images
testlabel  = mnist.test.labels
print (" type of 'trainimg' is %s"    % (type(trainimg)))
print (" type of 'trainlabel' is %s"  % (type(trainlabel)))
print (" type of 'testimg' is %s"     % (type(testimg)))
print (" type of 'testlabel' is %s"   % (type(testlabel)))
print (" shape of 'trainimg' is %s"   % (trainimg.shape,))
print (" shape of 'trainlabel' is %s" % (trainlabel.shape,))
print (" shape of 'testimg' is %s"    % (testimg.shape,))
print (" shape of 'testlabel' is %s"  % (testlabel.shape,))


# type of 'trainimg' is <type 'numpy.ndarray'>
# type of 'trainlabel' is <type 'numpy.ndarray'>
# type of 'testimg' is <type 'numpy.ndarray'>
# type of 'testlabel' is <type 'numpy.ndarray'>
# shape of 'trainimg' is (55000, 784)
# 784 = 28 * 28
# shape of 'trainlabel' is (55000, 10)
# shape of 'testimg' is (10000, 784)
# shape of 'testlabel' is (10000, 10)







"""
 How does the training data look like?
"""
print ("3. How does the data look like?")
nsample = 3
randidx = np.random.randint(trainimg.shape[0], size=nsample)

for i in randidx:
    curr_img   = np.reshape(trainimg[i, :], (28, 28)) # 28 by 28 matrix
    curr_label = np.argmax(trainlabel[i, :] ) # Label
    plt.matshow(curr_img, cmap=plt.get_cmap('gray'))
    plt.title("" + str(i) + "th Training Data " + "Label is " + str(curr_label))
    print ("" + str(i) + "th Training Data " + "Label is " + str(curr_label))










"""
Batch Learning?
"""
print ("4. Batch Learning? ")
batch_size = 100
batch_xs, batch_ys = mnist.train.next_batch(batch_size)
print ("type of 'batch_xs' is %s" % (type(batch_xs)))
print ("type of 'batch_ys' is %s" % (type(batch_ys)))
print ("shape of 'batch_xs' is %s" % (batch_xs.shape,))
print ("shape of 'batch_ys' is %s" % (batch_ys.shape,))







"""
 Get Random Batch with 'np.random.randint'
"""
print ("5. Get Random Batch with 'np.random.randint'")
randidx = np.random.randint(trainimg.shape[0], size=batch_size)
batch_xs2 = trainimg[randidx, :]
batch_ys2 = trainlabel[randidx, :]
print ("type of 'batch_xs2' is %s" % (type(batch_xs2)))
print ("type of 'batch_ys2' is %s" % (type(batch_ys2)))
print ("shape of 'batch_xs2' is %s" % (batch_xs2.shape,))
print ("shape of 'batch_ys2' is %s" % (batch_ys2.shape,))



plt.show()