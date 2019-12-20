#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/23 20:18
# @Author  : ganliang
# @File    : tf_perceptron.py
# @Desc    : 感知机y
import os

import numpy as np
import pandas
import tensorflow as tf
import tensorflow.contrib.layers as layers
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
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


def multiple_perceptron_mnist(epochs=10, batch_size=1000, learning_rate=0.01, hidden=30):
    """
    mnis手写数字识别
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


def multiple_perceptron_boston(epochs=10, learning_rate=0.001, batch_size=200, hidden=30):
    """
    使用多层感知机 预测波斯顿房价
    :return:
    """
    boston_data = datasets.load_boston()
    X_train, Y_train = boston_data.data, boston_data.target
    # 数据归一化
    # X_train = (X_train - np.mean(X_train)) / np.std(X_train)
    # Y_train = (Y_train - np.mean(Y_train)) / np.std(Y_train)
    minmax_scaler = MinMaxScaler()
    # 因为我们输出段使用的是sigmodid激活函数，要确保输出值在0,1范围之间。
    X_train = minmax_scaler.fit_transform(X_train)
    Y_train = minmax_scaler.fit_transform(np.reshape(Y_train, newshape=(-1, 1)))

    n_samples, n_feature = X_train.shape
    df = pandas.DataFrame(data=boston_data.data, columns=boston_data.feature_names)
    df["target"] = boston_data.target
    logger.info("\n%s" % df.describe(include="all"))

    X = tf.placeholder(dtype=tf.float32, shape=(None, n_feature), name="X")
    Y = tf.placeholder(dtype=tf.float32, name="Y")

    layer1 = layers.fully_connected(X, hidden, activation_fn=tf.nn.relu, scope="layer1")
    Y_hat = layers.fully_connected(layer1, 1, activation_fn=tf.nn.sigmoid, scope="layer2")

    with tf.name_scope("loss"):
        loss = tf.reduce_mean(tf.square(Y_hat - Y))
        tf.summary.scalar("loss", loss)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    summary_ops = tf.summary.merge_all()
    init = tf.global_variables_initializer()
    train_losss, test_losss = [], []
    with tf.Session() as sess:
        writer = tf.summary.FileWriter("multiple_perceptron_boston", graph=sess.graph)
        sess.run(init)
        for epoch in range(epochs):
            kfold = KFold(n_samples // batch_size, shuffle=True)
            total_train_loss, total_test_loss = 0, 0
            for train_index, test_index in kfold.split(X_train, Y_train):
                # 训练模型
                _, train_loss, summary_str = sess.run([optimizer, loss, summary_ops],
                                                      feed_dict={X: X_train[train_index], Y: Y_train[train_index]})
                writer.add_summary(summary_str, epoch)
                total_train_loss += train_loss

                # 测试模型损失
                test_loss = sess.run(loss, feed_dict={X: X_train[test_index], Y: Y_train[test_index]})
                total_test_loss += test_loss
            train_losss.append(total_train_loss / kfold.n_splits)
            test_losss.append(total_test_loss / kfold.n_splits)
            logger.info("epoch {0} train loss {1} test loss {2}".format(epoch, total_train_loss / kfold.n_splits,
                                                                        total_test_loss / kfold.n_splits))
        writer.close()

        # 计算下总的损失
        total_loss = sess.run(loss, feed_dict={X: X_train, Y: Y_train})
        logger.info("total loss {0}".format(total_loss))

        # 预测下房屋价格
        n = 500
        predict = sess.run(Y_hat, feed_dict={X: np.reshape(X_train[n], newshape=(1, -1)), Y: Y_train[n]})
        logger.info("pred {0}  real {1}".format(predict, Y_train[n]))

        predicts = sess.run(Y_hat, feed_dict={X: X_train, Y: Y_train})
        predicts = minmax_scaler.inverse_transform(predicts)
        logger.info("\n%s" % np.c_[predicts, boston_data.target])
        lines = list(range(len(boston_data.target)))
        plt.plot(lines, predicts, "ro", label="predict")
        plt.plot(lines, boston_data.target, "bo", label="real")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("loss train/test")
        plt.legend()
        plt.show()

        plt.plot(list(range(epochs)), train_losss, "ro", label="train_losss")
        plt.plot(list(range(epochs)), test_losss, "b", label="test_losss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("loss train/test")
        plt.legend()
        plt.show()
