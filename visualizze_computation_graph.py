
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# my imports
import utils

import tensorflow as tf

# Declaration phase

# Variables

# Constant gives shape=[row_n, col_n, ...] for build tensor
# specify dtype and name
a = tf.constant(2, name='a')
b = tf.constant(3, name='b')

# Operations
x = tf.add(a, b, name='add')

# my option
start = 3
limit = 18
delta = 3

r1 = tf.range(start, limit, delta)
r2 = tf.range(limit)


# Init debug tools
my_graph = tf.get_default_graph()

# Writer component for tensorboard
writer = tf.summary.FileWriter(utils.logdir, my_graph)

# Execution phase
with tf.Session() as sess:
    # Use sess.graph for get current graph
    print(sess.run(x))

# Close debug tools
writer.close()