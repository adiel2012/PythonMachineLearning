import tensorflow as tf
a = tf.constant(2)
b = tf.constant(3)
x = tf.add(a, b)
with tf.Session() as sess:
	# add this line to use TensorBoard. 
	writer = tf.summary.FileWriter('./graphs', sess.graph) 
	print( sess.run(x) )
writer.close() 


#	Go to terminal, run:
#	$ python [yourprogram].py
#	$ tensorboard --logdir="./graphs" --port 6006
#	Then open your browser and go to: http://localhost:6006/




a = tf.constant([2, 2], name="a")
b = tf.constant([[0, 1], [2, 3]], name="b")
x = tf.add(a, b, name="add")
y = tf.multiply(a, b, name="mul")
with tf.Session() as sess:
	x, y = sess.run([x, y]) 
	print (x, y) 

mm = tf.fill([2, 3], 8) # ==> [[8, 8, 8], [8, 8, 8]]
zz = tf.zeros([2, 3], tf.int32) #==> [[0, 0, 0], [0, 0, 0]]
with tf.Session() as sess:
	print (sess.run(zz)) 