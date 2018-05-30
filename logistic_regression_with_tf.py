
import os
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf

import utils

# Parameters
learn_rate = 0.01
batch_size = 128
n_epochs = 30
n_train = 60000
n_test = 10000

# Read data
print()
utils.download_mnist()
train, val, test = utils.read_mnist(flatten=True)
print()

# Create dataset with tf.data module
# https://www.tensorflow.org/performance/datasets_performance
train_data = tf.data.Dataset.from_tensor_slices(train)
train_data = train_data.shuffle(10000)
# combines executive elements of this dataset into batches
train_data = train_data.batch(batch_size)

test_data = tf.data.Dataset.from_tensor_slices(test)
test_data = test_data.batch(batch_size)


iterator = tf.data.Iterator.from_structure(train_data.output_types,
                                           train_data.output_shapes)

img, label = iterator.get_next()

train_init = iterator.make_initializer(train_data)
test_init = iterator.make_initializer(test_data)

# Create Weights and Bias
# w init to random value with mean = 0 and std = 0.01
#   for do that use tf.random_normal_initializer(mean, std)
# b init to 0
w = tf.get_variable("weights", shape=(784, 10), initializer=tf.random_normal_initializer(0, 0.01))
b = tf.get_variable("bias", shape=(1, 10), initializer=tf.zeros_initializer())

# build model
# later passed through softmax layer
logits = tf.matmul(img, w) + b

# Define loss function
entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=label, name="entropy")
# Computes the mean over all the examples in the batchs
loss = tf.reduce_mean(entropy, name="loss")


# Define training op
optimizer = tf.train.AdamOptimizer(learn_rate).minimize(loss)


# Calculate accuracy with test set
predictions = tf.nn.softmax(logits)
correct_predictions = tf.equal(tf.argmax(predictions, 1), tf.argmax(label, 1))

# parallel sum (reduce() is a parallel programming keyword)
accuracy = tf.reduce_sum(tf.cast(correct_predictions, tf.float32))


# EXECUTION PHASE

writer = tf.summary.FileWriter(utils.logdir, tf.get_default_graph())

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    # Train the model
    for i in range(n_epochs):
        # drawing samples from train_data
        sess.run(train_init) 

        tot_loss = 0
        n_batches = 0

        try:
            while True:
                _, loss_result = sess.run([optimizer, loss])
                tot_loss += loss_result
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        print('Average loss epoch {0}: {1}'.format(i, tot_loss / n_batches))
    #print('Total time: {0} seconds'.format(time.time() - start_time))

    # Test the model
    sess.run(test_init) 

    tot_correct_preds = 0
    try:
        while True:
            accuracy_batch = sess.run(accuracy)
            tot_correct_preds += accuracy_batch
    except tf.errors.OutOfRangeError:
        pass

    print('Acurracy {0}'.format(tot_correct_preds / n_test))

writer.close()