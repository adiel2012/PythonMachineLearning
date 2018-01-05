import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import sys

# https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/examples/tutorials/mnist/mnist_deep.py

def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""   
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
 
def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
strides=[1, 2, 2, 1], padding='SAME')
 
x_image = tf.placeholder(tf.float32, [None, 28, 28, 1])

# First convolutional layer - maps one grayscale image to 32 feature maps.
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

# Pooling layer - downsamples by 2X.
h_pool1 = max_pool_2x2(h_conv1)

with tf.Session() as sess:
  init = tf.global_variables_initializer()
  sess.run(init)
  lo = sess.run(h_pool1, feed_dict={x_image: np.random.randn(100, 28, 28, 1) })
  #print(lo.shape)
  img1 = lo[0,:,:,0]
  #print(img1.shape) 
  plt.imshow(img1, interpolation="nearest", cmap="gray")
  plt.show()