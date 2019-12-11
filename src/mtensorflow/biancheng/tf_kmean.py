#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/26 11:46
# @Author  : ganliang
# @File    : tf_kmean.py
# @Desc    : 聚类 无监督学习非常有用，因为现存的大多数数据是没有标签的，这种方法可以用于诸如模式识别、特征提取、数据聚类和降维等任务。

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.colors import ListedColormap
from sklearn import datasets
from tensorflow.contrib.learn import KMeansClustering

from src.config import logger


def kmean_iris_show():
    """
    聚类iris数据
    :return:
    """
    iris = datasets.load_iris()
    X, Y = iris.data[:, :2], iris.target
    plt.scatter(X[:, 0], X[:, 1])
    plt.xlabel("sepia length")
    plt.ylabel("sepia width")
    plt.show()


def kmean_iris_cluster():
    """
    聚类iris数据
    :return:
    """
    iris = datasets.load_iris()
    X, Y = iris.data[:, :2], iris.target

    def input_fn():
        return tf.constant(np.array(X), tf.float32, X.shape), None

    def ScatterPlot(X, Y, assignments=None, centers=None):
        if assignments is None:
            assignments = [0] * len(X)
        plt.figure(figsize=(14, 8))
        cmap = ListedColormap(["red", "green", "blue"])
        plt.scatter(X, Y, c=assignments, cmap=cmap)
        if centers is not None:
            plt.scatter(centers[:, 0], centers[:, 1], c=range(len(centers)), marker="+", s=400, cmap=cmap)
        plt.xlabel("sepia length")
        plt.ylabel("sepia width")
        plt.show()

    kmeans = KMeansClustering(num_clusters=3, relative_tolerance=0.0001, random_seed=2)
    kmeans.fit(input_fn=input_fn)
    clusters = kmeans.clusters()
    assignments = list(kmeans.predict_cluster_idx(input_fn=input_fn))

    ScatterPlot(X[:, 0], X[:, 1], assignments=assignments, centers=clusters)


def kmean_iris_distances(epochs=10):
    """
     # 平方误差和（SSE），随着簇数量 k 的增加，SSE 是逐渐减小的
    :return:
    """
    iris = datasets.load_iris()
    X, Y = iris.data[:, :2], iris.target

    def input_fn():
        return tf.constant(np.array(X), tf.float32, X.shape), None

    distances = []
    for epoch in range(1, epochs):
        kmeans = KMeansClustering(num_clusters=epoch, relative_tolerance=0.0001, random_seed=2)
        kmeans.fit(input_fn=input_fn)

        sum_distances = kmeans.score(input_fn=input_fn, steps=100)
        logger.info(sum_distances)
        distances.append(sum_distances)

    plt.plot(list(range(1, epochs)), distances)
    plt.xlabel("epoch")
    plt.ylabel("distance")
    plt.show()
