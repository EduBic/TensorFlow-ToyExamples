
import os
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import utils

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

DATA_FILE = 'data/birth_life_2010.csv'
max_epochs = 100

# Phase 1: Assemble the graph
# Step 1: read in data from the file
dataset = pd.read_csv(DATA_FILE)

def test_dataset():
	# see one row of dataset
	print(dataset.iloc[[0]])

	print(dataset.shape)
	for sample in dataset.itertuples(index=False, name="Pandas"):
		print(sample)


def huber_loss(labels, predictions, delta=14.0):
	residual = tf.abs(labels - predictions)
	def f1(): return 0.5 * tf.square(residual)
	def f2(): return delta * residual - 0.5 * tf.square(delta)
	
	return tf.cond(residual < delta, f1, f2)


# sheet = book.sheet_by_index(0)
# data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
# n_samples = sheet.nrows - 1

# Step 2: create placeholders for input X (number of fire) and label Y (number of theft)
X = tf.placeholder(dtype=tf.float32, name="X")
Y = tf.placeholder(dtype=tf.float32, name="Y")

# Step 3: create weight and bias, initialized to 0
w = tf.get_variable("weights", initializer=tf.constant(0.0))
b = tf.get_variable("bias", initializer=tf.constant(0.0))

# Step 4: predict Y (number of theft) from the number of fire
Y_predicted = w * X + b


# Step 5: use the square error as the loss function
loss = tf.square(Y_predicted - Y, name="loss")
loss = huber_loss(Y, Y_predicted)


# Step 6: using gradient descent to min loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)


# Phase 2: Train our model
writer = tf.summary.FileWriter(utils.logdir, tf.get_default_graph())

print()
with tf.Session() as sess:
	print()

	# Step 7: initialize the necessary variables w and b
	sess.run(tf.global_variables_initializer())

	# Step 8: train the model
	for epoch in range(max_epochs):
		for sample in dataset.itertuples(index=False):
			# Session runs optimizer to minimize loss 
			# and fetch the value of loss.

			x = sample.Birth_rate
			y = sample.Life_expectancy
			l = sess.run(optimizer, feed_dict={X: x, Y: y})

	w_out, b_out = sess.run([w, b])
	print("Weigth:", w_out)
	print("Bias:", b_out)

	print()

writer.close()

# plot the results
plt.plot(dataset['Birth_rate'], dataset['Life_expectancy'], 'bo', label="Real data")
plt.plot(dataset['Birth_rate'], dataset['Birth_rate'] * w_out + b_out, 'r', label='Predicted data')
plt.legend()
plt.show()