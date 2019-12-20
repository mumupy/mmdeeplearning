#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/24 9:03
# @Author  : ganliang
# @File    : tf_bpn.py
# @Desc    : tensorflow反向传播
import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

from src.config import logger, root_path


def _sigmaprime(X):
    return tf.multiply(tf.sigmoid(X), tf.subtract(tf.constant(1.0), tf.sigmoid(X)))


def _multilayer_perceptron(X, weights, biases):
    h_layer_1 = tf.add(tf.matmul(X, weights["h1"]), biases["h1"])
    out_layer_1 = tf.sigmoid(h_layer_1)

    h_out = tf.matmul(out_layer_1, weights["out"]) + biases["out"]
    return tf.sigmoid(h_out), h_out, out_layer_1, h_layer_1


def bpn(epochs=10, batch_size=1000, learning_rate=0.01, hidden=30):
    """
    反向传播实现
    :return:
    """
    mnist_path = os.path.join(root_path, "data", "fashionMNIST")
    mnist_data = input_data.read_data_sets(mnist_path, one_hot=True)
    train_data = mnist_data.train
    test_data = mnist_data.test

    sample_count = train_data.num_examples
    feature_count = train_data.images.shape[1]

    label_count = train_data.labels.shape[1]

    X = tf.placeholder(tf.float32, shape=(None, feature_count))
    Y = tf.placeholder(tf.float32, shape=(None, label_count))

    weights = {
        "h1": tf.Variable(tf.random_normal(shape=(feature_count, hidden), seed=0)),
        "out": tf.Variable(tf.random_normal(shape=(hidden, label_count), seed=0))
    }
    baises = {
        "h1": tf.Variable(tf.random_normal(shape=(1, hidden), seed=0)),
        "out": tf.Variable(tf.random_normal(shape=(1, label_count), seed=0))
    }
    Y_that, h_out, out_layer_1, h_layer_1 = _multilayer_perceptron(X, weights, baises)

    # 反向传播
    error = Y_that - Y
    delta_2 = tf.multiply(error, _sigmaprime(h_out))
    delta_w_2 = tf.matmul(tf.transpose(out_layer_1), delta_2)

    wtd_error = tf.matmul(delta_2, tf.transpose(weights["out"]))
    delta_1 = tf.multiply(wtd_error, _sigmaprime(h_layer_1))
    delta_w_1 = tf.matmul(tf.transpose(X), delta_1)

    eta = tf.constant(learning_rate)

    step = [tf.assign(weights["h1"], tf.subtract(weights["h1"], tf.multiply(eta, delta_w_1))),
            tf.assign(baises["h1"], tf.subtract(baises["h1"], tf.multiply(eta, tf.reduce_mean(delta_1, axis=[0])))),
            tf.assign(weights["out"], tf.subtract(weights["out"], tf.multiply(eta, delta_w_2))),
            tf.assign(baises["out"], tf.subtract(baises["out"], tf.multiply(eta, tf.reduce_mean(delta_2, axis=[0]))))]

    acct_mat = tf.equal(tf.argmax(Y_that, 1), tf.argmax(Y, 1))
    with tf.name_scope("accuracy"):
        accuracy = tf.reduce_mean(tf.cast(acct_mat, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

    init = tf.global_variables_initializer()
    summary_ops = tf.summary.merge_all()
    acc_trains, acc_tests = [], []
    with tf.Session() as sess:
        sess.run(init)
        writer = tf.summary.FileWriter("bpn", graph=sess.graph)
        for i in range(epochs):
            batch_count = sample_count // batch_size
            for j in range(batch_count):
                batch_trains, batch_lables = mnist_data.train.next_batch(batch_size)
                _, summary_str = sess.run([step, summary_ops], feed_dict={X: batch_trains, Y: batch_lables})
                writer.add_summary(summary_str, i * batch_size + j)
            # 训练数据评估值
            acc_train = sess.run(accuracy, feed_dict={X: train_data.images, Y: train_data.labels})
            # 测试数据评估值
            acc_test = sess.run(accuracy, feed_dict={X: test_data.images, Y: test_data.labels})
            logger.info("epoll {0} train accuracy {1} test accuracy {2}".format(i, acc_train, acc_test))
            acc_trains.append(acc_train)
            acc_tests.append(acc_test)
        writer.close()

    plt.plot(list(range(epochs)), acc_trains, "bo", label="train accuracy")
    plt.plot(list(range(epochs)), acc_tests, "r", label="test accuracy")
    plt.xlabel("epoch")
    plt.xlabel("accuracy")
    plt.title("accuracy train/test")
    plt.legend()
    plt.show()
