# Basics - constants

import tensorflow as tf

# Creating constant nodes
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)
print(node1, node2)

# Create a session and run
sess = tf.Session()
print(sess.run([node1, node2]))

# Hello world node
node3 = tf.constant("Hello world", dtype=tf.string)
print(sess.run(node3))

#Sum node
node4 = tf.add(node1, node2)
print("Sum node value: {}".format(sess.run(node4)))