# -*- coding: utf-8 -*-
__author__ = 'whale'
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

"""
 Basic TensorFlow
 Sungjoon Choi (sungjoon.choi@cpslab.snu.ac.kr)
"""
import numpy as np
import tensorflow as tf

hello = tf.constant("Hello, it's me.")
print hello
# Tensor("Const:0", shape=(), dtype=string)
"""
 This will not show "Hellos, it's me."
"""

"""
 In order to make things happen, we need 'session'!
"""
sess = tf.Session()

"""
 Run session with tf variable
"""
hello_out = sess.run(hello)
print "Type of 'hello' is ", type(hello)
print "Type of 'hello_out' is ", type(hello_out)
print hello_out

"""
 Until you run session, nothing happens!
"""
print "Until you run session, nothing happens!"

"""
 There are other types as well
  1. Constant types
  2. Operators
  # 연산할 것들. 컨벌루션, 렉티파이 리니어
  3. Variables
  # 파라미터
  4. Placeholder (Buffers)
  # 데이터가 지나갈 통로
"""

"""
 Constant types
"""
print "\nConstant types (numpy)"
a = tf.constant(1.5)
b = tf.constant(2.5)
print " 'a': ", a, " Type is ", type(a)
print " 'b': ", b, " Type is ", type(b)
a_out = sess.run(a)
b_out = sess.run(b)
print " Type of 'a_out' is ", type(a_out)
print " Type of 'b_out' is ", type(b_out)
print " a_out is ", a_out, "b_out is ", b_out, "a_out+b_out is ", a_out+b_out

"""
 Operators are also tf variables
"""
print "\nOperators (tf.add, tf.mul)"
add = tf.add(a, b)
print " 'add' is ", add, ' type is ', type(add)
add_out = sess.run(add)
print " 'add_out' is ", add_out, ' type is ', type(add_out)
mul = tf.mul(a, b)
print " 'mul' is ", mul, ' type is ', type(mul)
mul_out = sess.run(mul)
print " 'mul_out' is ", mul_out, ' type is ', type(mul_out)

"""
 Variables & PlaceHolder
"""
print "\nVariables & PlaceHolders"
X = np.random.rand(1, 20)
Input  = tf.placeholder(tf.float32, [None, 20])

# 안쪽에 네트워크.
# 20바이10 짜리 메트릭스
# 일종의 레이어
Weight = tf.Variable(tf.random_normal([20, 10], stddev=0.5))

# 아웃풋 계위를 맞추기 위해서 bias를 사용하는 거다
# 선형 문제에서는 절편.
# 데이터는 10 ~ 20 에서 찰박찰박 노는데 아웃풋이 1000 ~ 1500 이면 이 안으로 넣기 위해서 사용.
Bias   = tf.Variable(tf.zeros([1, 10]))
print " 'Weight': ", Weight, " Type is ", type(Weight)
print " 'Bias': ", Bias, " Type is ", type(Bias)
# Weight_out = sess.run(Weight) # <= This is not allowed!
# print Weight.eval(sess) # <= This is not also allowed! (Do You Know Why??)

"""
 Initialize Variables
"""
print "\nInitialize Variables"
# 이거 하면 이니셜라이징 하라고 명령만 하는거 아직 안채움.
init = tf.initialize_all_variables()
sess.run(init)
print " 'Weight': ", Weight, " Type is ", type(Weight)
print " 'Bias': ", Bias, " Type is ", type(Bias)
print Weight.eval(sess)

"""
 Operations with Variables and PlaceHolders
"""
print "\nOperations with Variables and PlaceHolders"
oper = tf.matmul(Input, Weight) + Bias
val  = sess.run(oper, feed_dict={Input:X})
print " oper is ", oper, " type is ", type(oper)
print " val is ", val, " type is ", type(val)

"""
 Operators with PlaceHolder
 (This is very important !)
 (Remember 'feed_dict' !!!)
"""
print "\nOperators with PlaceHolder (tf.add, tf.mul)"
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
add = tf.add(x, y)
mul = tf.mul(x, y)
add_out = sess.run(add, feed_dict={x:5.0, y:6.0})
mul_out = sess.run(mul, feed_dict={x:5.0, y:6.0})
print " addres: ", add_out, " type is ", type(add_out)
print " mulres: ", mul_out, " type is ", type(mul_out)
