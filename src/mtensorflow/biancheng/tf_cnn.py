#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/25 9:42
# @Author  : ganliang
# @File    : tf_cnn.py
# @Desc    : cnn卷积神经网络
import os

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from src.config import logger, root_path


def mnist_conv2d(epochs=10, learning_rate=0.001, batch_size=200, dropout=0.85):
    """
    手写图像识别
    :param epochs:
    :param learning_rate:
    :param batch_size:
    :param dropout:
    :return:
    """
    mnist = input_data.read_data_sets(os.path.join(root_path, "data", "fashionMNIST"), one_hot=True)
    num_examples, num_feature = mnist.train.images.shape  # 550000 784
    num_labels = mnist.train.labels.shape[1]  # 10

    X = tf.placeholder(dtype=tf.float32, shape=(None, num_feature))
    Y = tf.placeholder(dtype=tf.float32, shape=(None, num_labels))
    keep_prob = tf.placeholder(dtype=tf.float32)

    def conv2d(X, w, b, strides=1):
        X = tf.nn.conv2d(X, w, strides=[1, strides, strides, 1], padding="SAME")
        X = tf.nn.bias_add(X, b)
        return tf.nn.relu(X)

    def maxpool2d(X, k=2):
        return tf.nn.max_pool(X, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding="SAME")

    # 定义卷积
    def conv_net(X, weights, biases, dropout):
        X = tf.reshape(X, shape=(-1, 28, 28, 1))
        conv1 = conv2d(X, weights["wc1"], biases["bc1"])
        conv1 = maxpool2d(conv1, k=2)

        conv2 = conv2d(conv1, weights["wc2"], biases["bc2"])
        conv2 = maxpool2d(conv2, k=2)

        # 全连接层
        fc1 = tf.reshape(conv2, [-1, weights["wd1"].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights["wd1"]), biases["bd1"])
        fc1 = tf.nn.relu(fc1)

        fc1 = tf.nn.dropout(fc1, dropout)
        out = tf.add(tf.matmul(fc1, weights["out"]), biases["out"])
        return out

    weights = {
        "wc1": tf.Variable(tf.random_normal([5, 5, 1, 32])),
        "wc2": tf.Variable(tf.random_normal([5, 5, 32, 64])),
        "wd1": tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
        "out": tf.Variable(tf.random_normal([1024, num_labels])),
    }
    biases = {
        "bc1": tf.Variable(tf.random_normal([32])),
        "bc2": tf.Variable(tf.random_normal([64])),
        "bd1": tf.Variable(tf.random_normal([1024])),
        "out": tf.Variable(tf.random_normal([num_labels])),
    }

    Y_hat = conv_net(X, weights, biases, keep_prob)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y_hat, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Y_hat, 1), tf.argmax(Y, 1)), tf.float32))

    init = tf.global_variables_initializer()

    train_losss, train_accuracys, test_accuracys = [], [], []
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(epochs):
            batch_count = num_examples // batch_size
            total_loss, batch_x, batch_y = 0, None, None
            for i in range(batch_count):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                _, l = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: dropout})
                total_loss += l
            train_accuracy = sess.run(accuracy, feed_dict={X: batch_x, Y: batch_y, keep_prob: 1})
            test_accuracy = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1})
            logger.info(
                "epoch {0} loss {1} test accuracy {2} train accuracy {3}".format(epoch, total_loss / batch_count,
                                                                                 test_accuracy, train_accuracy))
            train_losss.append(total_loss / batch_count)
            test_accuracys.append(test_accuracy)
            train_accuracys.append(train_accuracy)

    plt.plot(train_losss, "k-", label="test accuracy")
    plt.show()

    plt.plot(list(range(epochs)), test_accuracys, "k-", label="test accuracy")
    plt.plot(list(range(epochs)), train_accuracys, "r--", label="train accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("accuracy test/train")
    plt.legend()
    plt.show()
