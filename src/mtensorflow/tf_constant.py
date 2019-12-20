#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/21 16:15
# @Author  : ganliang
# @File    : tf_constant.py
# @Desc    : tf的常量使用方式

import os

import numpy as np
import tensorflow as tf
from src.config import logger

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def helloworld():
    message = tf.constant("hello world")
    with tf.Session() as sess:
        logger.info(sess.run(message).decode())


def add():
    a = tf.constant(1, np.float32)
    b = tf.constant(2, np.float32)
    c = tf.add(a, b)
    with tf.Session() as sess:
        logger.info(sess.run(c))


def interactive_add():
    sess = tf.InteractiveSession()
    a = tf.constant(1, np.float32)
    b = tf.constant(2, np.float32)

    c = tf.add(a, b)
    logger.info(c.eval())
    logger.info(a.eval())
    sess.close()


def constant():
    """
    常量
    :return:
    """
    a = tf.constant(1)
    b = tf.constant([1, 2, 3])
    c = tf.constant(0, dtype=tf.float32, shape=[3, 3])
    z = tf.constant(1, dtype=tf.float32, shape=[3, 3])
    with tf.Session() as sess:
        logger.info(sess.run(a))
        logger.info(sess.run(b))
        logger.info(sess.run(c))
        logger.info(sess.run(z))


def zero_onse():
    c = tf.constant(0, dtype=tf.float32, shape=[3, 3])
    cc = tf.zeros(shape=[3, 3], dtype=tf.float32)
    cl = tf.zeros_like(c)
    zz = tf.ones(dtype=tf.float32, shape=[3, 3])
    ol = tf.ones_like(zz)
    with tf.Session() as sess:
        logger.info(sess.run(c))
        logger.info(sess.run(cc))
        logger.info(sess.run(cl))
        logger.info(sess.run(zz))
        logger.info(sess.run(ol))


def linspace():
    """
    等量补差
    :return:
    """
    l = tf.linspace(1., 10., 100)
    with tf.Session() as sess:
        logger.info(sess.run(l))


def range():
    l = tf.range(1, 10, 2, dtype=tf.float32)
    with tf.Session() as sess:
        logger.info(sess.run(l))


def random():
    """
    求随机值
    :return:
    """
    tf.set_random_seed(100)

    # 均值（默认值=0.0）和标准差（默认值=1.0）、形状为 [M，N] 的正态分布随机数组：
    n = tf.random_normal(shape=(3, 3), mean=0.0, stddev=1.0)
    tn = tf.truncated_normal(shape=(3, 3), mean=0.0, stddev=1.0)

    # 要在种子的[minval（default = 0），maxval] 范围内创建形状为[M，N] 的给定伽马分布随机数组
    u = tf.random_uniform(shape=(3, 3), minval=1, maxval=10)

    g = tf.random_gamma(shape=(3, 3), alpha=1)

    # 数据截取
    corp = tf.random_crop(g, size=(3, 2))

    # 对数据进行混洗
    sg = tf.random_shuffle(g)
    with tf.Session() as sess:
        logger.info("random_normal\n %s" % sess.run(n))
        logger.info("truncated_normal\n %s" % sess.run(tn))
        logger.info("random_uniform\n %s" % sess.run(u))
        logger.info("random_gamma\n %s" % sess.run(g))
        logger.info("random_crop\n %s" % sess.run(corp))
        logger.info("random_shuffle\n %s" % sess.run(sg))
