
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import utils

import tensorflow as tf

# Declaration phase

# Defiine a placeholder
# tf.placeholder(dtype, shape=None, name=None)

# placeholder for a vector of 3 elements
a = tf.placeholder(tf.float32, shape=[3], name="a_placeholder")
b = tf.constant([5, 5, 5], tf.float32, name="const_b")
op = a + b

add_op = tf.add(2, 5)
mul_op = tf.multiply(add_op, 3)

writer = tf.summary.FileWriter(utils.logdir, tf.get_default_graph())

print()
with tf.Session() as sess:
    print()

    # give a placeholder for a
    if tf.get_default_graph().is_feedable(a): # check if placeholder
        print(sess.run(op, feed_dict={a : [1,2,3]}))

    print(sess.run(mul_op)) # >> 21
    print(sess.run(mul_op, feed_dict={add_op: 15})) # >> 45

    print()

writer.close()