import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# See https://www.tensorflow.org/versions/r0.8/tutorials/mnist/beginners/ for
# details. This is where all this code comes from.

# Import the dataset.
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# x is a 2D matrix that represents image data.
# D0 = index of the image (we process many at a time)
# D1 = index of the pixel (they are flattened)
# V = value of the pixel (between 0 and 1)
x = tf.placeholder(tf.float32, [None, 784])

# W is a 2D matrix that contains the weights used to predict the evidence.
# D0 = index of the pixel
# D1 = index of the class (= digit)
# V = weight
W = tf.Variable(tf.zeros([784, 10]))

# b is the bias for each class.
b = tf.Variable(tf.zeros([10]))

# y is the softmax of the evidence. Therefore its shape is [N, 10] and the sum of
# all 10 values is always 1.
# D0 = index of the image
# D1 = index of the class (= digit)
# V = probability of the image belonging to that class
y = tf.nn.softmax(tf.matmul(x, W) + b)

# y_ will contain the correct answers. This will help us calculate the
# cross-entropy. It looks similar to y, except that it contains exactly one 1
# and the rest is filled with 0s.
# D0 = index of the image
# D1 = index of the class (= digit)
# V = 1 if image is of this class, 0 otherwise
y_ = tf.placeholder(tf.float32, [None, 10])

# Cross-entropy. See http://colah.github.io/posts/2015-09-Visual-Information/
cross_entropy = tf.reduce_mean(
    -tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# TensorFlow is smart enough to adjust our variables W and b to minimize
# cross-entropy. Cool stuff.
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# 1D boolean tensor of prediction correctness.
# - D0 = index of the image
# - V = whether digit was correctly labeled
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# This will be a number between 0 and 1.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={
            x: batch_xs,
            y_: batch_ys
        })
        print(sess.run(accuracy, feed_dict={
            x: mnist.test.images,
            y_: mnist.test.labels
        }))
