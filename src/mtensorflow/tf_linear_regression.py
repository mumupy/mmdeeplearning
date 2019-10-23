#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/23 10:43
# @Author  : ganliang
# @File    : tf_linear_regression.py
# @Desc    : 单特征的线性回归
import tensorflow as tf
from tensorflow.contrib.learn.python.learn import datasets
import numpy as np
from matplotlib import pyplot as plt

from src.config import logger


def normaize(X):
    """
    数据归一化处理
    :param X:
    :return:
    """
    return (X - np.mean(X)) / np.std(X)


def loaddata():
    """
    获取到数据集
    :return:
    """
    boston_data = datasets.load_boston()
    X_train, Y_train = boston_data.data[:, 5], boston_data.target
    # X_train=normaize(X_train)
    n_samples = len(X_train)
    return (X_train, Y_train, n_samples)


def get_model():
    """
    定义模型
    :return:
    """
    # 定义X,Y占位符
    X = tf.placeholder(dtype=tf.float32, name="X")
    Y = tf.placeholder(dtype=tf.float32, name="Y")

    # 定义权重和偏差
    w = tf.Variable(0.0, dtype=tf.float32, name="w")
    b = tf.Variable(0.0, dtype=tf.float32, name="b")

    return (X, Y, w, b)


def get_loss(X, Y, w, b):
    """
    定义损失函数
    :return:
    """
    # 线性回归模型
    Y_that = X * w + b
    # 平方损失
    loss = tf.square(Y - Y_that, name="loss")
    return loss


def get_optimizer(loss):
    """
    获取优化方法
    :param loss:
    :return:
    """
    # 随机梯度下降优化算法
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
    # optimizer = tf.train.AdagradOptimizer(learning_rate=0.01).minimize(loss)
    return optimizer


def show_data(X_train, Y_train, w, b):
    """
    显示真实数据和预测数据之间的差异
    :param X_train: 训练数据
    :param Y_train: 标签
    :param w: 权重值
    :param b: 偏差值
    :return:
    """
    y_pred = X_train * w + b

    # 对比预估值和真实值
    plt.plot(X_train, Y_train, "bo", label="Real Data")
    plt.plot(X_train, y_pred, "r", label="predict Data")
    plt.legend()
    plt.show()


def show_loss(total):
    """
    展示损失函数变化图
    :return:
    """
    plt.plot(total)
    plt.show()


def linear_regression():
    """
    简单的线性回归模型
    :return:
    """
    X, Y, w, b = get_model()
    loss = get_loss(X, Y, w, b)
    optimizer = get_optimizer(loss)

    linear_ops = tf.global_variables_initializer()

    X_train, Y_train, n_samples = loaddata()
    total = []
    with tf.Session() as sess:
        sess.run(linear_ops)
        writer = tf.summary.FileWriter("linear_regression", sess.graph)
        for i in range(100):
            total_loss = 0
            for x, y in zip(X_train, Y_train):
                _, l = sess.run([optimizer, loss], feed_dict={X: x, Y: y})
                total_loss += l
            total.append(total_loss / n_samples)
            logger.info("epoll {0} loss {1}".format(i, total_loss / n_samples))
        writer.close()
        w_val, b_val = sess.run([w, b])
        logger.info("w {0},b {1}".format(w_val, b_val))
        writer.close()

        show_data(X_train, Y_train, w_val, b_val)

        # 查看损失变化图
        show_loss(total)


def k_linear_regression():
    """
    k折交叉验证发
    :return:
    """
    from sklearn.model_selection import KFold
    X, Y, w, b = get_model()
    loss = get_loss(X, Y, w, b)
    optimizer = get_optimizer(loss)

    linear_ops = tf.global_variables_initializer()

    X_train, Y_train, n_samples = loaddata()
    total = []
    n_splits = 5

    with tf.Session() as sess:
        sess.run(linear_ops)
        writer = tf.summary.FileWriter("linear_regression", sess.graph)
        for i in range(10):
            total_train_loss = 0
            total_test_loss = 0

            k_fold = KFold(n_splits=n_splits, shuffle=False, random_state=None)
            for train_index, test_index in k_fold.split(X_train, Y_train):
                # 训练数据
                for x, y in zip(X_train[train_index], Y_train[train_index]):
                    _, l = sess.run([optimizer, loss], feed_dict={X: x, Y: y})
                    total_train_loss += l / len(train_index)

                # 测试数据 计算评估误差
                w_val, b_val = sess.run([w, b])
                test_loss = get_loss(X_train[test_index], Y_train[test_index], w_val, b_val)
                total_test_loss += tf.reduce_mean(test_loss).eval()

            total.append(total_train_loss / n_splits)
            logger.info("epoll {0} train loss {1} test loss {2}".format(i, total_train_loss / n_splits,
                                                                        total_test_loss / n_splits))
        writer.close()
        w_val, b_val = sess.run([w, b])
        logger.info("w {0},b {1}".format(w_val, b_val))
        writer.close()

        # 对比预估值和真实值
        show_data(X_train, Y_train, w_val, b_val)

        # 查看损失变化图
        show_loss(total)


def multiple_linear_regression():
    """
    多特征的线性回归模型
    :return:
    """
    boston_data = datasets.load_boston()
    X_train, Y_train = boston_data.data, boston_data.target
    n_samples = len(X_train)

    # 将偏差作为一列特征添加进去
    X_train = np.c_[X_train, np.zeros(shape=(n_samples, 1))]
    n_features = X_train.shape[1]

    X_train = (X_train - np.mean(X_train)) / np.std(X_train)
    Y_train = np.reshape(Y_train, newshape=(n_samples, 1))
    # TODO 暂时不考虑偏差，考虑偏差就是把偏差也当做一个特征值

    X = tf.placeholder(dtype=tf.float32, shape=(n_samples, n_features), name="X")
    Y = tf.placeholder(dtype=tf.float32, shape=(n_samples, 1), name="Y")

    w = tf.Variable(tf.random_normal(shape=(n_features, 1)), name="w")

    Y_that = tf.matmul(X, w)
    loss = tf.reduce_mean(tf.square(Y - Y_that), name="loss")
    optimizer = tf.train.GradientDescentOptimizer(0.001, name="optimizer").minimize(loss)

    linear_ops = tf.global_variables_initializer()
    total = []
    with tf.Session() as sess:
        sess.run(linear_ops)
        writer = tf.summary.FileWriter("multiplinear_regression", sess.graph)
        for i in range(1000):
            _, total_loss = sess.run([optimizer, loss], feed_dict={X: X_train, Y: Y_train})
            total.append(total_loss)
            logger.info("epoll {0} loss {1}".format(i, total_loss / n_samples))
        writer.close()
        w_value = sess.run(w)
        writer.close()

        show_loss(total)
        # 预测
        n = 500
        Y_pred = np.matmul(X_train[n, :], w_value)
        logger.info("pred {0} real {1}".format(Y_pred[0], Y_train[n][0]))
