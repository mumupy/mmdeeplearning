#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/19 21:53
# @Author  : ganliang
# @File    : mnist.py
# @Desc    : mnist数字识别

import os

import tensorflow as tf

from src.config import logger
from src.config import root_path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def mnist_simple():
    from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

    y_ = tf.placeholder("float", [None, 10])

    logger.info(y_)

    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    x = tf.placeholder("float", [None, 784])
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)

    dataset = read_data_sets(os.path.join(root_path, "data", "fashionMNIST"))
    for i in range(1000):
        batch_xs, batch_ys = dataset.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print(sess.run(accuracy, feed_dict={x: dataset.images, y_: dataset.labels}))


if __name__ == "__main__":
    mnist_simple()
