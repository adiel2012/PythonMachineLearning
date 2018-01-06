import tensorflow as tf
import numpy as np

# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/linear_regression.py
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/multilayer_perceptron.py

n_pattern = 51
n_hidden = 9
n_column = 22
n_classes = 3

# Parameters
learning_rate = 0.01
training_epochs = 10000
display_step = 50;

# this is the data to train
x = np.random.randn(n_pattern, n_column)
y = np.random.randn(n_pattern, n_classes)

x_placeholder = tf.placeholder(tf.float32, [None, n_column])
y_placeholder = tf.placeholder(tf.float32, [None, n_classes])

W = tf.Variable(tf.zeros([n_column, n_hidden]))
b_W = tf.Variable(tf.zeros([n_hidden]))
hidden_ouput = tf.nn.sigmoid(tf.matmul(x_placeholder, W) + b_W)
B = tf.Variable(tf.zeros([n_hidden, n_classes]))
b_o = tf.Variable(tf.zeros([n_classes]))
output = tf.nn.sigmoid(tf.matmul(hidden_ouput, B) + b_o)

# Mean squared error
cost = tf.reduce_sum(tf.pow(output-y_placeholder, 2))/(2*n_pattern)
# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(init)
  # Training cycle
  for epoch in range(training_epochs):
    _, c = sess.run([optimizer, cost], feed_dict={x_placeholder: x , y_placeholder: y})
        
    if epoch % display_step == 0:
      print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(c))
    
  print("Optimization Finished!")  
  print(sess.run(output, feed_dict={x_placeholder: x}))