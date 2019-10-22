#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/22 11:26
# @Author  : ganliang
# @File    : tf_matrix.py
# @Desc    : tensorflow矩阵

import tensorflow as tf

from src.config import logger


def matrix_matmul():
    """
    矩阵相乘  Ci,j= Ai,k*Bk,j+....
    :return:
    """
    isses = tf.InteractiveSession()
    A = tf.Variable(tf.eye(4))
    B = tf.Variable(tf.random_normal(shape=(4, 4)))

    A.initializer.run()
    B.initializer.run()

    logger.info("A\n%s" % A.eval())
    logger.info("B\n%s" % B.eval())

    logger.info("tf.matmul(A,B)\n%s" % tf.matmul(A, B).eval())
    isses.close()


def matrix_transpose():
    """
    矩阵转置,将行转成列
    :return:
    """
    isses = tf.InteractiveSession()
    A = tf.Variable(tf.random_normal(shape=(4, 4)))
    A.initializer.run()

    logger.info("A\n%s" % A.eval())
    logger.info("tf.transpose(A)\n%s" % tf.transpose(A).eval())

    # 在最后两个维度进行置换
    B = tf.Variable(tf.random_normal(shape=(1, 2, 3, 4)))
    B.initializer.run()
    logger.info("B\n%s" % B.eval())
    logger.info("tf.matrix_transpose(A)\n%s" % tf.matrix_transpose(B).eval())
    isses.close()


def matrix_trace():
    """
    矩阵轨迹
    :return:
    """
    isses = tf.InteractiveSession()
    A = tf.Variable(tf.random_normal(shape=(4, 4)))
    A.initializer.run()

    logger.info("A\n%s" % A.eval())
    logger.info("tf.trace(A)\n%s" % tf.trace(A).eval())
    isses.close()


def matrix_determinant():
    """
    计算方阵行列式的值
    :return:
    """
    isses = tf.InteractiveSession()
    A = tf.Variable(tf.random_normal(shape=(4, 4)))
    A.initializer.run()

    logger.info("A\n%s" % A.eval())
    logger.info("tf.matrix_determinant(A)\n%s" % tf.matrix_determinant(A).eval())
    isses.close()


def matrix_inverse():
    """
    求解可逆方阵的逆
    :return:
    """
    isses = tf.InteractiveSession()
    A = tf.Variable(tf.random_normal(shape=(4, 4)))
    A.initializer.run()

    logger.info("A\n%s" % A.eval())
    logger.info("tf.matrix_inverse(A)\n%s" % tf.matrix_inverse(A).eval())
    isses.close()


def matrix_svd():
    """
    奇异值分解
    :return:
    """
    isses = tf.InteractiveSession()
    A = tf.Variable(tf.random_normal(shape=(4, 4)))
    A.initializer.run()

    logger.info("A\n%s" % A.eval())
    logger.info("tf.svd(A)\n {0}".format(tf.svd(A)))
    isses.close()


def matrix_qr():
    """
    qr分解
    :return:
    """
    isses = tf.InteractiveSession()
    A = tf.Variable(tf.random_normal(shape=(4, 4)))
    A.initializer.run()

    logger.info("A\n%s" % A.eval())
    logger.info("tf.qr(A)\n {0}".format(tf.qr(A)))
    isses.close()


def matrix_diag():
    """
    qr分解
    :return:
    """
    isses = tf.InteractiveSession()
    # 对角值
    a = tf.constant([1, 2, 3, 4])

    logger.info("a\n%s" % a.eval())
    logger.info("tf.diag(a)\n {0}".format(tf.diag(a).eval()))
    isses.close()


def matrix_reduce():
    """
    reduce
    :return:
    """
    isses = tf.InteractiveSession()
    # 对角值
    X = tf.constant([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]])

    logger.info("X\n%s" % X.eval())
    logger.info("tf.reduce_sum(X)\n {0}".format(tf.reduce_sum(X).eval()))
    logger.info("tf.reduce_sum(X,axis=0)\n {0}".format(tf.reduce_sum(X, axis=0).eval()))
    logger.info("tf.reduce_sum(X,axis=1)\n {0}".format(tf.reduce_sum(X, axis=1).eval()))

    logger.info("tf.reduce_mean(X)\n {0}".format(tf.reduce_mean(X).eval()))
    logger.info("tf.reduce_mean(X,axis=0)\n {0}".format(tf.reduce_mean(X, axis=0).eval()))
    logger.info("tf.reduce_mean(X,axis=1)\n {0}".format(tf.reduce_mean(X, axis=1).eval()))

    logger.info("tf.reduce_max(X)\n {0}".format(tf.reduce_max(X).eval()))
    logger.info("tf.reduce_max(X,axis=0)\n {0}".format(tf.reduce_max(X, axis=0).eval()))
    logger.info("tf.reduce_max(X,axis=1)\n {0}".format(tf.reduce_max(X, axis=1).eval()))

    logger.info("tf.reduce_min(X)\n {0}".format(tf.reduce_min(X).eval()))
    logger.info("tf.reduce_min(X,axis=0)\n {0}".format(tf.reduce_min(X, axis=0).eval()))
    logger.info("tf.reduce_min(X,axis=1)\n {0}".format(tf.reduce_min(X, axis=1).eval()))
    isses.close()


def matrix_cumsum():
    """
    :return:
    """
    isses = tf.InteractiveSession()
    # 对角值
    X = tf.constant([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]])

    logger.info("X\n%s" % X.eval())
    logger.info("tf.cumsum(X)\n {0}".format(tf.cumsum(X).eval()))
    logger.info("tf.cumsum(X,exclusive=True)\n {0}".format(tf.cumsum(X, exclusive=True).eval()))
    logger.info("tf.cumsum(X,reverse=True)\n {0}".format(tf.cumsum(X, reverse=True).eval()))
    logger.info(
        "tf.cumsum(X,exclusive=True,reverse=True)\n {0}".format(tf.cumsum(X, exclusive=True, reverse=True).eval()))
    isses.close()


def matrix_segment():
    """
    :return:
    """
    isses = tf.InteractiveSession()
    # 对角值
    X = tf.constant([5., 1., 7., 2., 3., 4., 1., 3.], dtype=tf.float32)
    s_id = [0, 0, 0, 1, 2, 2, 3, 3]

    logger.info("X\n%s" % X)
    logger.info("s_id\n%s" % s_id)
    logger.info("tf.segment_sum(X,s_id)\n {0}".format(tf.segment_sum(X, s_id)))
    logger.info("tf.segment_mean(X,s_id)\n {0}".format(tf.segment_mean(X, s_id).eval()))
    logger.info("tf.segment_max(X, s_id)\n {0}".format(tf.segment_max(X, s_id).eval()))
    logger.info("tf.segment_min(X, s_id)\n {0}".format(tf.segment_min(X, s_id).eval()))
    logger.info("tf.segment_prod(X, s_id)\n {0}".format(tf.segment_prod(X, s_id).eval()))
    logger.info("tf.unsorted_segment_sum(X, s_id)\n {0}".format(tf.unsorted_segment_sum(X, s_id, 2)))

    # c = tf.constant([0., 1.], dtype=tf.float32)
    # logger.info("tf.sparse_segment_sum(X, s_id)\n {0}".format(tf.sparse_segment_sum(X, c, s_id)))
    # logger.info("tf.sparse_segment_mean(X, s_id)\n {0}".format(tf.sparse_segment_mean(X, c, s_id)))
    # logger.info("tf.sparse_segment_sqrt_n(X, s_id)\n {0}".format(tf.sparse_segment_sqrt_n(X, c, s_id)))
    isses.close()


def matrix_condition():
    """
    :return:
    """
    isses = tf.InteractiveSession()
    # 对角值
    X = tf.constant([5., 1., 7., 2., 3., 4., 1., 3.], dtype=tf.float32)
    logger.info("X\n%s" % X.eval())
    logger.info("tf.argmax(X)\n {0}".format(tf.argmax(X).eval()))
    logger.info("tf.argmin(X)\n {0}".format(tf.argmin(X).eval()))
    logger.info("tf.unique(X)\n {0}".format(tf.unique(X)))
    isses.close()
