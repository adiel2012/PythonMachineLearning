import tensorflow as tf
a = tf.add(3,5)
print (a)  # Tensor("Add:0", shape=(), dtype=int32)

sess = tf.Session()
print( sess.run(a) )    #8
sess.close() 

#More Graph
x = 2
y = 3
op1 = tf.add(x,y)
op2 = tf.multiply(x,y)
op3 = tf.pow(op1,op2)
with tf.Session() as sess2:
	op3 = sess2.run(op3)
	print(op3)
	
	
# Creates a graph.
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name='b')
c = tf.multiply(a, b)
with tf.Session() as sess:
	res = sess.run(c)
	print(res)
	
	

	
#More Graph
#to add operators to a graph, set it as default:
#g = tf.Graph()
#with g.as_default():
#	x = tf.add(3, 5)
#sess = tf.Session(graph=g)
#with tf.Session() as sess:
#	sess.run(x)