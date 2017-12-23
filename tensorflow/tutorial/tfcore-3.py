# Variables and models
import tensorflow as tf

# Declaring variables with initial values and type
var1 = tf.Variable([0.3], dtype=tf.float32)
var2 = tf.Variable([-0.3], dtype=tf.float32)

# Placeholder
testVar = tf.placeholder(tf.float32)

sess = tf.Session()

# For initilalizing global variables
init = tf.global_variables_initializer()
sess.run(init)

# Creating test model
linear_model = var1 * testVar + var2

print("Linear model: ", sess.run(linear_model, {testVar: [1, 2, 3, 4]}))

# For holding expected values
expected_values = tf.placeholder(tf.float32)

# Square of difference between actual and expected values
squared_deltas = tf.square(linear_model - expected_values)
# Calculate deviation
loss = tf.reduce_sum(squared_deltas)

# Print the loss value
print("Loss value: ", sess.run(loss, {testVar: [1, 2, 3, 4], expected_values: [0, -1, -2, -3]}))

# Reassign and retest
fix_var1 = tf.assign(var1, [-1.])
fix_var2 = tf.assign(var2, [1.])
sess.run([fix_var1, fix_var2])

# Print the loss value
print("Loss value after correction: ", sess.run(loss, {testVar: [1, 2, 3, 4], expected_values: [0, -1, -2, -3]}))

# ==================================================================================================================
# ==================================================================================================================

# Training models

# Create an optimizer to minimize loss
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# Reset values
sess.run(init)

# Run 1000 times
for i in range(1000):
    sess.run(train, {testVar: [1, 2, 3, 4], expected_values: [0, -1, -2, -3]})

print("Values after training: ", sess.run([var1, var2]))

# Print the loss value
print("Loss value after training: ", sess.run(loss, {testVar: [1, 2, 3, 4], expected_values: [0, -1, -2, -3]}))


