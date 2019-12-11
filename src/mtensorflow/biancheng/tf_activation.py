#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/23 20:52
# @Author  : ganliang
# @File    : tf_activation.py
# @Desc    : tensorflow激活函数

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.config import logger


def activation_threhold():
    """
    阈值激活函数,这是最简单的激活函数。在这里，如果神经元的激活值大于零，那么神经元就会被激活；否则，它还是处于抑制状态。下面绘制阈值激活函数的图，随着神经元的激活值的改变
    :return:
    """

    def threhold(X):
        cond = tf.less(X, tf.zeros(shape=tf.shape(X), dtype=X.dtype))
        out = tf.where(cond, tf.zeros(tf.shape(X)), tf.ones(tf.shape(X)))
        return out

    h = np.linspace(-1., 1., 50)
    logger.info("h\n{0}".format(h))
    out = threhold(h)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        y = sess.run(out)

        plt.title("threhold activation")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.plot(h, y)
        plt.show()


def activation_sigmoid():
    """
    Sigmoid 激活函数：在这种情况下，神经元的输出由函数 g(x)=1/(1+exp(-x)) 确定。在 TensorFlow 中，方法是 tf.sigmoid，它提供了 Sigmoid 激活函数。这个函数的范围在 0 到 1 之间：
    :return:
    """

    h = np.linspace(-10, 10, 50)
    out = tf.sigmoid(h)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        y = sess.run(out)

        plt.title("sigmoid activation")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.plot(h, y)
        plt.show()


def activation_tanh():
    """
    在数学上，它表示为 (1-exp(-2x)/(1+exp(-2x)))。在形状上，它类似于 Sigmoid 函数，但是它的中心位置是 0，其范围是从 -1 到 1。TensorFlow 有一个内置函数 tf.tanh，用来实现双曲正切激活函数：
    :return:
    """

    h = np.linspace(-10, 10, 50)
    out = tf.tanh(h)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        y = sess.run(out)

        plt.title("tanh activation")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.plot(h, y)
        plt.show()


def activation_liner():
    """
    线性激活函数：在这种情况下，神经元的输出与神经元的输入值相同。这个函数的任何一边都不受限制
    :return:
    """
    h = np.linspace(-10., 10., 30)

    w = tf.Variable(tf.random_normal(shape=(3, 1), stddev=2, dtype=tf.float64))
    b = tf.Variable(tf.random_normal(shape=(1,), stddev=2, dtype=tf.float64))
    liner_out = tf.matmul(h.reshape(10, 3), w) + b

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        y = sess.run(liner_out)

        plt.title("liner activation")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.plot(np.linspace(-10., 10., len(y)), y.reshape(len(y), ))
        plt.show()


def activation_relu():
    """
    线性激活函数：在这种情况下，神经元的输出与神经元的输入值相同。这个函数的任何一边都不受限制
    :return:
    """
    h = np.linspace(-10., 10., 30)
    out = tf.nn.relu(h)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        y = sess.run(out)

        plt.title("relu activation")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.plot(h, y)
        plt.show()

def activation_softmax():
    """
    Softmax 激活函数是一个归一化的指数函数。一个神经元的输出不仅取决于其自身的输入值，还取决于该层中存在的所有其他神经元的输入的总和。这样做的一个优点是使得神经元的输出小，因此梯度不会过大。数学表达式为 yi =exp(xi​)/Σjexp(xj)：
    :return:
    """
    h = np.linspace(-10., 10., 30)
    out = tf.nn.softmax(h)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        y = sess.run(out)

        plt.title("softmax activation")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.plot(h, y)
        plt.show()
