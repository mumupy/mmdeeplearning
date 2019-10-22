#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/21 17:28
# @Author  : ganliang
# @File    : tf_variable.py
# @Desc    : tf变量，变量声明之后必须初始化，在神经网络中变量一般主要用于权重和偏差

import tensorflow as tf
from src.config import logger
import numpy as np


def variable():
    """
    变量必须初始化
    :return:
    """
    v = tf.Variable(tf.random_normal(shape=(3, 3)))

    # 初始化 所有的变量
    initial_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(initial_op)
        logger.info(v)


def saver():
    v = tf.Variable(tf.random_gamma(shape=(3, 3), alpha=10))
    initial_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(initial_op)
        logger.info(v)
        saver = tf.train.Saver()
        saver.save(sess, save_path="./saver")


def restore():
    v = tf.Variable(tf.random_gamma(shape=(3, 3), alpha=10))
    initial_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(initial_op)
        logger.info(v)
        saver = tf.train.Saver()
        saver.restore(sess, save_path="./saver")


def placeholder():
    ph = tf.placeholder(dtype=tf.float32, shape=(3, 4))
    with tf.Session() as sess:
        logger.info("placeholder\n%s" % sess.run(ph, feed_dict={ph: np.arange(12).reshape(3, 4)}))


def convert_to_tensor():
    ct = tf.convert_to_tensor(np.arange(12).reshape(3, 4), dtype=np.float32)
    with tf.Session() as sess:
        logger.info("convert_to_tensor \n %s"%sess.run(ct))

def range():
    range=tf.range(12)
    with tf.Session() as sess:
        logger.info("range \n%s"%sess.run(range))