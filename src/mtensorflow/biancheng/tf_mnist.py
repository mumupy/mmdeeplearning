#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/19 21:53
# @Author  : ganliang
# @File    : mnist.py
# @Desc    : mnist数字识别

import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from src.config import root_path, logger

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def mnist_simple():
    # 加载数据
    mnist_data = input_data.read_data_sets(os.path.join(root_path, "data", "fashionMNIST"), one_hot=True)
    train_data = mnist_data.train

    # 权重和偏置
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    # 占位符
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])

    # 损失函数
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    loss = tf.reduce_mean(tf.square(y_-y))
    # loss = -tf.reduce_sum(y_ * tf.log(y))
    # loss = tf.losses.softmax_cross_entropy(y_, y)
    # loss = tf.losses.sparse_softmax_cross_entropy(y_,y)

    # 优化函数
    # optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    optimizer = tf.train.AdadeltaOptimizer(0.001).minimize(loss)

    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(10000):
            batch_xs, batch_ys = train_data.next_batch(100)
            _, lloss = sess.run([optimizer, loss], feed_dict={x: batch_xs, y_: batch_ys})
            logger.info(lloss)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        logger.info(sess.run(accuracy, feed_dict={x: train_data.images, y_: train_data.labels}))


if __name__ == "__main__":
    mnist_simple()
