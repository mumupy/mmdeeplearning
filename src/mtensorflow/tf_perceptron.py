#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/23 20:18
# @Author  : ganliang
# @File    : tf_perceptron.py
# @Desc    : 感知机
import os

import tensorflow as tf
import tensorflow.contrib.layers as layers
from matplotlib import pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

from src.config import logger, root_path


def threhold(X):
    cond = tf.less(X, tf.zeros(shape=tf.shape(X), dtype=X.dtype))
    out = tf.where(cond, tf.zeros(tf.shape(X)), tf.ones(tf.shape(X)))
    return out


def perceptron():
    """
    当只有一层这样的神经元存在时，它被称为感知机
    :return:
    """
    learn_rate = 0.4
    epsilon = 1e-03
    max_epochs = 1000

    T, F = 1., 0.
    X_in = [[T, T, T, T], [T, T, F, T], [T, F, T, T], [T, F, F, T],
            [F, T, T, T], [F, T, F, T], [F, F, T, T], [F, F, F, T]]
    Y = [[T], [T], [F], [F], [T], [F], [F], [F]]

    W = tf.Variable(tf.random_normal(shape=(4, 1), stddev=2))
    h = tf.matmul(X_in, W)
    # Y_hat = threhold(h)
    Y_hat = tf.sigmoid(h)

    error = Y - Y_hat
    mean_error = tf.reduce_mean(tf.square(error))
    dW = learn_rate * tf.matmul(X_in, error, transpose_a=True)
    # 更新权重
    train = tf.assign(W, W + dW)

    init = tf.global_variables_initializer()
    epoch = 0
    err = 1
    with tf.Session() as sess:
        sess.run(init)
        opoches, errs, = [], []
        while err > epsilon and epoch < max_epochs:
            epoch += 1
            _, err = sess.run([train, mean_error])
            logger.info("epoch {0} mean error {1}".format(epoch, err))
            opoches.append(epoch)
            errs.append(err)

        plt.title("perceptron epoch loss")
        plt.xlabel("opoch")
        plt.ylabel("error")
        plt.plot(opoches, errs)
        plt.show()


def multiple_perceptron(epochs=10, batch_size=1000, learning_rate=0.01, hidden=30):
    """
    输入层被称为第零层，因为它只是缓冲输入。存在的唯一一层神经元形成输出层。输出层的每个神经元都有自己的权重和阈值。当存在许多这样的层时，网络被称为多层感知机（MLP）。MLP有一个或多个隐藏层。这些隐藏层具有不同数量的隐藏神经元。每个隐藏层的神经元具有相同的激活函数：
    MLP 也被称为全连接层。MLP 中的信息流通常是从输入到输出，目前没有反馈或跳转，因此这些网络也被称为前馈网络。
    :return:
    """
    mnist_path = os.path.join(root_path, "data", "fashionMNIST")
    mnist_data = input_data.read_data_sets(mnist_path, one_hot=True)
    train_data = mnist_data.train
    test_data = mnist_data.test

    sample_count = train_data.num_examples  # 55000
    feature_count = train_data.images.shape[1]  # 784

    label_count = train_data.labels.shape[1]  # 10

    X = tf.placeholder(dtype=tf.float32, shape=(None, feature_count), name="X")
    Y = tf.placeholder(dtype=tf.float32, shape=(None, label_count), name="Y")

    with tf.name_scope("layer"):
        layer1 = layers.fully_connected(X, hidden, activation_fn=tf.nn.relu, scope="layer1")
        layer2 = layers.fully_connected(layer1, 256, activation_fn=tf.nn.relu, scope="layer2")
        Y_that = layers.fully_connected(layer2, label_count, activation_fn=None, scope="layerout")
    with tf.name_scope("cross_entropy"):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=Y_that, name="loss"))
        tf.summary.scalar("cross_entropy", loss)
    with tf.name_scope("train"):
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    init = tf.global_variables_initializer()
    summary_ops = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(init)
        writer = tf.summary.FileWriter("multiple_perceptron", graph=sess.graph)
        total = []
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

        plt.plot(total)
        plt.show()
