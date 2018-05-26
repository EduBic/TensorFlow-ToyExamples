
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import utils

import tensorflow as tf

# print out the graph's protobuf
def print_graph_protobuf():
    const = tf.constant([1.0, 2.0], name="a_const")
    print(tf.get_default_graph().as_graph_def())

# print_graph_protobuf()

# use x.initializer for init
# x.value() to get the value
# x.assign(..) to write value
s_bad = tf.Variable(2, name="simple_scalar")
m_bad = tf.Variable([[0, 1], [2, 3]], name="matrix")
W_bad = tf.Variable(tf.zeros([784, 10], name="big_matrix"))

# A more good practice use tf.get_variable
s = tf.get_variable("simple_scalar", initializer=tf.constant(2))
m = tf.get_variable("matrix", initializer=tf.constant([[0,1], [2,3]]))
W = tf.get_variable("big_matrix", shape=(784, 10), initializer=tf.zeros_initializer())

# initialize with random value
V = tf.get_variable("normal_mat", shape=(784, 10), 
                    initializer=tf.truncated_normal_initializer()) 

# Try the assignment
w = tf.get_variable("assing_var", initializer=tf.constant(10))
assign_op = w.assign(w * 10)

writer = tf.summary.FileWriter(utils.logdir, tf.get_default_graph())

print("")

with tf.Session() as sess:
    print("") # for better read terminal
    
    # Variables are not init yet
    print(sess.run(tf.report_uninitialized_variables()))

    # fetch an initializer operation
    sess.run(tf.global_variables_initializer())
    # OR initialize only the vars you want with variables_initializer([a,b])
    # OR use var.initializer for each variables

    # Evaluate a variable
    print(sess.run(V))
    # OR
    print(V.eval())

    print("")

    # Assignment
    print(W.eval()) # >> 10
    sess.run(assign_op)
    print(w.eval()) # >> 100
    sess.run(assign_op)
    print(w.eval()) # >> 1000

    print(sess.run(w.assign_add(42))) # >>
    print(sess.run(w.assign_sub(42))) # >>

    print("")

writer.close()