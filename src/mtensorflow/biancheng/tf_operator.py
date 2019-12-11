#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/22 9:28
# @Author  : ganliang
# @File    : tf_operator.py
# @Desc    : tensorflow基本操作符

import tensorflow as tf

from src.config import logger


def matrix_operator():
    isess = tf.InteractiveSession()
    X = tf.Variable(tf.eye(3))
    W = tf.Variable(tf.random_normal(shape=(3, 3)))

    X.initializer.run()
    W.initializer.run()
    logger.info("X\n%s" % X.eval())
    logger.info("W\n%s" % W.eval())

    logger.info("tf.add(X, W)\n%s" % tf.add(X, W).eval())
    logger.info("tf.subtract(X, W)\n%s" % tf.subtract(X, W).eval())
    logger.info("tf.mod(X, W)\n%s" % tf.mod(X, W).eval())
    logger.info("tf.divide(X, W)\n%s" % tf.divide(X, W).eval())
    logger.info("tf.matmul(W,X)\n%s" % tf.matmul(W, X).eval())
    # 相同位置元素进行相乘
    logger.info("tf.multiply(W,X)\n%s" % tf.multiply(W, X).eval())
    isess.close()


def matrix_power():
    isess = tf.InteractiveSession()
    X = tf.Variable(tf.eye(3))
    W = tf.Variable(tf.random_normal(shape=(3, 3)))

    X.initializer.run()
    W.initializer.run()
    logger.info("X\n%s" % X.eval())
    logger.info("W\n%s" % W.eval())

    logger.info("tf.pow(X, W)\n%s" % tf.pow(X, W).eval())
    logger.info("tf.square(W)\n%s" % tf.square(W).eval())
    logger.info("tf.sqrt(W)\n%s" % tf.sqrt(W).eval())
    logger.info("tf.exp(W)\n%s" % tf.exp(W).eval())
    logger.info("tf.log(W)\n%s" % tf.log(W).eval())
    isess.close()


def matrix_abs():
    isess = tf.InteractiveSession()
    X = tf.Variable(tf.eye(3))
    W = tf.Variable(tf.random_normal(shape=(3, 3)))

    X.initializer.run()
    W.initializer.run()
    logger.info("X\n%s" % X.eval())
    logger.info("W\n%s" % W.eval())
    # 取负
    logger.info("tf.negative(X)\n%s" % tf.negative(X).eval())
    # 返回 x 的符号
    logger.info("tf.sign(W)\n%s" % tf.sign(W).eval())
    logger.info("tf.reciprocal(W)\n%s" % tf.reciprocal(W).eval())
    logger.info("tf.abs(W)\n%s" % tf.abs(W).eval())
    logger.info("tf.round(W)\n%s" % tf.round(W).eval())

    # 向上取整
    logger.info("tf.ceil(W)\n%s" % tf.ceil(W).eval())

    # 向下取整
    logger.info("tf.floor(W)\n%s" % tf.floor(W).eval())

    # 取最接近的整数
    logger.info("tf.rint(W)\n%s" % tf.rint(W).eval())
    logger.info("tf.maximum(W)\n%s" % tf.maximum(W, X).eval())
    logger.info("tf.minimum(W)\n%s" % tf.minimum(W, X).eval())
    isess.close()


def matrix_sin():
    isess = tf.InteractiveSession()
    W = tf.Variable(tf.random_normal(shape=(3, 3)))

    W.initializer.run()
    logger.info("W\n%s" % W.eval())

    logger.info("tf.cos(W)\n%s" % tf.cos(W).eval())
    logger.info("tf.sin(W)\n%s" % tf.sin(W).eval())
    logger.info("tf.tan(W)\n%s" % tf.tan(W).eval())
    logger.info("tf.acos(W)\n%s" % tf.acos(W).eval())
    logger.info("tf.asin(W)\n%s" % tf.asin(W).eval())
    logger.info("tf.atan(W)\n%s" % tf.atan(W).eval())
    isess.close()


def matrix_other():
    isess = tf.InteractiveSession()
    X = tf.Variable(tf.eye(3))
    W = tf.Variable(tf.random_normal(shape=(3, 3)))

    X.initializer.run()
    W.initializer.run()
    logger.info("X\n%s" % X.eval())
    logger.info("W\n%s" % W.eval())

    logger.info("tf.div(X,W)\n%s" % tf.div(X, W).eval())
    logger.info("tf.truediv(X,W)\n%s" % tf.truediv(X, W).eval())
    logger.info("tf.floordiv(X,W)\n%s" % tf.floordiv(X, W).eval())
    logger.info("tf.realdiv(X,W)\n%s" % tf.realdiv(X, W).eval())

    # logger.info("tf.truncatediv(X,W)\n%s" % tf.truncatediv(X, W).eval())
    logger.info("tf.floor_div(X,W)\n%s" % tf.floor_div(X, W).eval())
    logger.info("tf.truncatemod(X,W)\n%s" % tf.truncatemod(X, W).eval())
    logger.info("tf.floormod(X,W)\n%s" % tf.floormod(X, W).eval())

    logger.info("tf.cross(X,W)\n%s" % tf.cross(X, W).eval())
    logger.info("tf.add_n(X,W)\n%s" % tf.add_n([X, W]).eval())
    logger.info("tf.squared_difference(X,W)\n%s" % tf.squared_difference(X, W).eval())

    isess.close()
