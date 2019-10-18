#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/17 17:13
# @Author  : ganliang
# @File    : perceptron.py
# @Desc    : 感知机模型
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

from src.config import logger


def get_train_data():
    """
    我们用的数据集是iris数据集，这个数据集是用来给花做分类的数据集，每个样本包括了花萼长度、花萼宽度、花瓣长度、花瓣宽度四个特征，通过这四个特征来将花分为山鸢尾、变色鸢尾还是维吉尼亚鸢尾。本次我只抽取一部分数据做二分类。
    :return:
    """
    iris = load_iris()
    raw_data = iris.data
    data = pd.DataFrame(raw_data, columns=iris.feature_names)
    data['label'] = iris.target

    data_array = np.array(data.iloc[:100, [0, 1, -1]])
    X, Y = data_array[:, :-1], data_array[:, -1]
    Y = np.array([1 if i == 1 else -1 for i in Y])  # 将标签为0,1，变为-1，+1
    return X, Y


def show_data():
    X, Y = get_train_data()
    plt.scatter(X[:50, 0], X[:50, 1], label='0')
    plt.scatter(X[50:100, 0], X[50:100, 1], label='1')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend()
    plt.show()


def plt_perceptron():
    # 定义sign函数
    def sign(X, W, b):
        return np.dot(W, X) + b

    # 遍历数据集
    def train(X, Y, W, b, learning_rate=0.1):
        for i in range(len(X)):
            if (Y[i] * sign(X[i], W, b) <= 0):
                W = W + learning_rate * Y[i] * X[i]
                b = b + learning_rate * Y[i]
        return W, b

    # 将数据集遍历1000遍，每100次打印一下W, b值
    W = np.zeros([1, 2])
    b = 0
    X, Y = get_train_data()
    for i in range(1000):
        W, b = train(X, Y, W, b)
        if (i % 100 == 0): logger.info("count={0} w={1} b={2}".format(i, W, b))

    x_ = np.linspace(4, 7, 10)
    y_ = -(W[0][0] * x_ + b) / W[0][1]
    plt.plot(x_, y_)
    plt.scatter(X[:50, 0], X[:50, 1], label='0')
    plt.scatter(X[50:100, 0], X[50:100, 1], label='1')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    plt_perceptron()
