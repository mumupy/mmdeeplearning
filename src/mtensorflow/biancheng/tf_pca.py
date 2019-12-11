#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/26 10:28
# @Author  : ganliang
# @File    : tf_pca.py
# @Desc    : 主成分分析（Principal Component Analysis，PCA）是一种多变量统计方法，它是最常用的降维方法之一，通过正交变换将一组可能存在相关性的变量数据转换为一组线性不相关的变量，转换后的变量被称为主成分。

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.examples.tutorials.mnist import input_data

from src.config import logger, root_path


class TF_PCA(object):

    def __init__(self, data, dtype=tf.float32):
        self._data = data
        self._dtype = dtype
        self._graph = None
        self._X = None
        # sigma 矩阵、正交矩阵 U 和奇异值
        self._u = None
        self._singular_values = None
        self._sigma = None

    def fit(self):
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._X = tf.placeholder(self._dtype, shape=self._data.shape)
            singular_values, u, _ = tf.svd(self._X)
            segma = tf.diag(singular_values)
        with tf.Session(graph=self._graph) as sess:
            self._u, self._singular_values, self._sigma = sess.run([u, singular_values, segma],
                                                                   feed_dict={self._X: self._data})

    def reduct(self, n_dimensions=None, keep_info=None):
        if keep_info:
            normalized_singular_values = self._singular_values / sum(self._singular_values)
            info = np.cumsum(normalized_singular_values)
            it = iter(idx for idx, value in enumerate(info) if value >= keep_info)
            n_dimensions = next(it) + 1
        with self._graph.as_default():
            sigma = tf.slice(self._sigma, [0, 0], [self._data.shape[1], n_dimensions])
            pca = tf.matmul(self._u, sigma)
        with tf.Session(graph=self._graph) as sess:
            return sess.run(pca, feed_dict={self._X: self._data})


def pca_basic():
    """
    :return:
    """
    mnist = input_data.read_data_sets(os.path.join(root_path, "data", "fashionMNIST"))
    tf_pca = TF_PCA(data=mnist.train.images)
    tf_pca.fit()
    pca = tf_pca.reduct(keep_info=0.15)
    logger.info("original data shape {0}".format(mnist.train.images.shape))
    logger.info("reduce data shape {0}".format(pca.shape))

    Set = sns.color_palette("Set2", 10)
    color_mapping = {key: value for (key, value) in enumerate(Set)}
    colors = list(map(lambda x: color_mapping[x], mnist.train.labels))
    ax = Axes3D(plt.figure())
    ax.scatter(np.reshape(pca[:, 0], newshape=(1, -1)), np.reshape(pca[:, 1], newshape=(1, -1)),
               np.reshape(pca[:, 2], newshape=(1, -1)), np.reshape(pca[:, 3], newshape=(1, -1)), c=colors)
    plt.show()


def pca_embeddings():
    """

    :return:
    """
    from tensorflow.contrib.tensorboard.plugins import projector
    mnist = input_data.read_data_sets(os.path.join(root_path, "data", "fashionMNIST"))
    images = tf.Variable(mnist.test.images, name="images")

    log_dir = "pca_embeddings"
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    metadata_path = "metadata.csv"

    with open(os.path.join(log_dir, metadata_path), "w") as metadata_file:
        for row in mnist.test.labels:
            metadata_file.write("%d\n" % row)
    with tf.Session() as sess:
        saver = tf.train.Saver([images])
        sess.run(images.initializer)
        saver.save(sess, os.path.join(log_dir, "images.ckpt"))

        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = images.name
        embedding.metadata_path = metadata_path
        projector.visualize_embeddings(tf.summary.FileWriter(log_dir),config)
