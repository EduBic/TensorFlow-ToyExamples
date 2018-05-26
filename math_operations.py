
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import utils

import tensorflow as tf

# tf datatype
t0 = 42  # 0-d tensor -> a scalar
t0_0 = tf.zeros_like(t0, name="t0_0")  # value = 0
t0_1 = tf.ones_like(t0, name="t0_1")   # value = 0

t1 = [0, 2, 4] # 1-d tensor -> a vector
t1_0 = tf.zeros_like(t1, name="t1_0")  # value = [b'', b'', b'']
t1_1 = tf.ones_like(t1, name="t1_1")  

t2 = [[True, False, False],
      [False, False, True],
      [False, True, False]] # 2-d tensor -> a matrix
t2_0 = tf.zeros_like(t2, name="t2_0")    # 3x3 tensor, all elem = False
t2_1 = tf.ones_like(t2, name="t2_1")     # 3x3 tensor, all elem = True


#
a = tf.constant([2, 2], name='a')
b = tf.constant([[0,1], [2,3]], name='b')

c = tf.constant([10, 20], name='c')
d = tf.constant([2, 3], name='d')

ops = [
    # division
    tf.div(b, a),
    tf.divide(b, a),
    tf.truediv(b, a),
    tf.floordiv(b, a),
    # tf.realdiv(b, a) # nly for real value
    tf.truncatediv(b, a),

    # moltiplication
    tf.multiply(c, d), # element-wise
    tf.tensordot(c, d, 1), # output 1 value

    tf.add(t1_0, t1_1)
]

writer = tf.summary.FileWriter(utils.logdir, tf.get_default_graph())

with tf.Session() as sess:
    # run() output is a numpy array
    [print(sess.run(op)) for op in ops]
    
writer.close()