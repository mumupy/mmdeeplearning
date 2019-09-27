#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/27 9:05
# @Author  : ganliang
# @File    : convolutionallayer.py
# @Desc    : 卷积层 卷积神经网络（convolutional neural network）是含有卷积层（convolutional layer）的神经网络

from mxnet import autograd, nd
from mxnet.gluon import nn

from src.config import logger


class Conv2D(nn.Block):
    def __init__(self, kernel_size, **kwargs):
        super(Conv2D, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=kernel_size)
        self.bias = self.params.get('bias', shape=(1,))

    def forward(self, x):
        return corr2d(x, self.weight.data()) + self.bias.data()


def corr2d(X, K):  # 本函数已保存在d2lzh包中方便以后使用
    h, w = K.shape
    Y = nd.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y


def cal_corr2d():
    X = nd.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    K = nd.array([[0, 1], [2, 3]])
    Y = corr2d(X, K)
    logger.info("卷积核:%s" % Y)


def convolutional():
    X = nd.ones((6, 8))
    X[:, 2:6] = 0

    conv2d = nn.Conv2D(1, kernel_size=(1, 2))
    conv2d.initialize()

    # 二维卷积层使用4维输入输出，格式为(样本, 通道, 高, 宽)，这里批量大小（批量中的样本数）和通
    # 道数均为1
    X = nd.ones((6, 8))
    X[:, 2:6] = 0
    K = nd.array([[1, -1]])
    Y = corr2d(X, K)
    logger.info("卷积核: %s" % Y)

    X = X.reshape((1, 1, 6, 8))
    Y = Y.reshape((1, 1, 6, 7))

    for i in range(10):
        with autograd.record():
            Y_hat = conv2d(X)
            l = (Y_hat - Y) ** 2
        l.backward()
        # 简单起见，这里忽略了偏差
        conv2d.weight.data()[:] -= 3e-2 * conv2d.weight.grad()
        if (i + 1) % 2 == 0:
            logger.info('batch %d, loss %.3f' % (i + 1, l.sum().asscalar()))

    logger.info(conv2d.weight.data().reshape((1, 2)))


def comp_conv2d(conv2d, X):
    conv2d.initialize()
    # (1, 1)代表批量大小和通道数（“多输入通道和多输出通道”一节将介绍）均为1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:])  # 排除不关心的前两维：批量和通道


if __name__ == "__main__":
    # cal_corr2d()
    convolutional()

    conv2d = nn.Conv2D(1, kernel_size=3, padding=1)
    X = nd.random.uniform(shape=(8, 8))
    print(comp_conv2d(conv2d, X).shape)
