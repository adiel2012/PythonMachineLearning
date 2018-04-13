import tensorflow as tf
import numpy as np

# https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/examples/tutorials/mnist/mnist_deep.py

num_channels = 3;

x_image = tf.placeholder(tf.float32, [None, 152, 152, num_channels])

# First convolutional layer
W_conv1 = tf.Variable(tf.truncated_normal([11, 11, num_channels, 32], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
h_conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='VALID') + b_conv1
h_conv1_relu = tf.nn.relu(h_conv1)
C1 = tf.nn.max_pool(h_conv1_relu, ksize=[1, num_channels, num_channels, 1], strides=[1, 2, 2, 1], padding='SAME', name = "C1")

print(C1)

W_conv2 = tf.Variable(tf.truncated_normal([9, 9, 32, 16], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[16]))
h_conv2 = tf.nn.conv2d(C1, W_conv2, strides=[1, 1, 1, 1], padding='VALID') + b_conv2
C3 = tf.nn.relu(h_conv2, name="C3")
#C3 = tf.nn.max_pool(h_conv2_relu, ksize=[1, num_channels, num_channels, 1], strides=[1, 2, 2, 1], padding='SAME')

print(C3)







 # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    #fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    #fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    #fc1 = tf.nn.relu(fc1)