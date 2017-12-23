# Mathematical operations

import tensorflow as tf

# Variable nodes and an adder
node1 = tf.placeholder(tf.int16)
node2 = tf.placeholder(tf.int16)
adder_node = node1 + node2

# Run adder with dictionary values
sess = tf.Session()
print(sess.run(adder_node, {node1: 5, node2: 6}))
print(sess.run(adder_node, {node1: [3, 4], node2: [12, 13]}))

# Triple the sum
add_and_triple = adder_node * 3
print(sess.run(add_and_triple, {node1: 3, node2: 2}))
print(sess.run(add_and_triple, {node1: [10, 12], node2: [5, 7]}))