#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/23 16:42
# @Author  : ganliang
# @File    : tf_logistic_regression.py
# @Desc    : 逻辑回归
import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import mnist, input_data
import numpy as np
from matplotlib import pyplot as plt

from src.config import logger, root_path


def logistic_regression(epochs=10, batch_size=1000, learning_rate=0.01):
    """
    逻辑回归基本实现
    :return:
    """
    mnist_path = os.path.join(root_path, "data", "fashionMNIST")
    mnist_data = input_data.read_data_sets(mnist_path, one_hot=True)
    train_data = mnist_data.train
    test_data = mnist_data.test

    sample_count = train_data.num_examples
    feature_count = train_data.images.shape[1]

    label_count = train_data.labels.shape[1]

    X = tf.placeholder(dtype=tf.float32, shape=(None, feature_count), name="X")
    Y = tf.placeholder(dtype=tf.float32, shape=(None, label_count), name="Y")

    w = tf.Variable(tf.zeros(shape=(feature_count, label_count)), name="w")
    b = tf.Variable(tf.zeros(shape=(label_count,)), name="b")

    tf.summary.histogram("weights", w)
    tf.summary.histogram("biases", b)

    with tf.name_scope("wx_b"):
        Y_that = tf.nn.softmax(tf.matmul(X, w) + b)
    with tf.name_scope("cross_entropy"):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=Y_that, name="loss"))
        tf.summary.scalar("cross_entropy", loss)
    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    init = tf.global_variables_initializer()
    summary_ops = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(init)
        writer = tf.summary.FileWriter("logistic_regression", graph=sess.graph)
        total=[]
        for i in range(epochs):
            total_loss = 0
            batch_count = sample_count // batch_size
            for j in range(batch_count):
                batch_trains, batch_lables = mnist_data.train.next_batch(batch_size)
                _, l, summary_str = sess.run([optimizer, loss, summary_ops],
                                             feed_dict={X: batch_trains, Y: batch_lables})
                writer.add_summary(summary_str, i * batch_size + j)
                total_loss += l
            logger.info("epoll {0} loss {1}".format(i, total_loss / batch_count))
            total.append(total_loss / batch_count)
        writer.close()

        # 模型评估
        correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_that, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))
        logger.info("accuracy %s" % sess.run(accuracy, feed_dict={X: test_data.images, Y: test_data.labels}))

        # 模型预测
        n = 50000
        w_value, b_value = sess.run([w, b])
        n_image = np.reshape(train_data.images[n], newshape=(1, feature_count))
        pred = tf.nn.softmax(tf.matmul(n_image, w_value) + b_value).eval()
        logger.info("\npred: {0} \nreal: {1}".format(pred[0], train_data.labels[n]))
        logger.info(np.c_[pred[0], train_data.labels[n]])

        plt.plot(total)
        plt.show()