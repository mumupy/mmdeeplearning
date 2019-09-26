#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/24 11:09
# @Author  : ganliang
# @File    : multilayerperceptron.py
# @Desc    : 多层感知机。
# 多层感知机就是含有至少一个隐藏层的由全连接层组成的神经网络，且每个隐藏层的输出通过激活函数进行变换。多层感知机的层数和各隐藏层中隐藏单元个数
# 都是超参数。以单隐藏层为例并沿用本节之前定义的符号，多层感知机按以下方式计算输出：

import d2lzh as d2l
from mxnet import nd, init, gluon
from mxnet.gluon import loss as gloss, nn


def basic_multilayer():
    """
    基本的多层感知机实现
    :return:
    """

    def relu(X):
        return nd.maximum(X, 0)

    def net(X):
        X = X.reshape((-1, num_inputs))
        H = relu(nd.dot(X, W1) + b1)
        return nd.dot(H, W2) + b2

    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    num_inputs, num_outputs, num_hiddens = 784, 10, 256

    W1 = nd.random.normal(scale=0.01, shape=(num_inputs, num_hiddens))
    b1 = nd.zeros(num_hiddens)
    W2 = nd.random.normal(scale=0.01, shape=(num_hiddens, num_outputs))
    b2 = nd.zeros(num_outputs)
    params = [W1, b1, W2, b2]

    for param in params:
        param.attach_grad()

    loss = gloss.SoftmaxCrossEntropyLoss()

    num_epochs, lr = 5, 0.5
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)


def simple_multilayer():
    """
    多层感知机简洁实现
    :return:
    """
    net = nn.Sequential()
    net.add(nn.Dense(256, activation='relu'), nn.Dense(10))
    net.add(gluon.nn.Dropout(0.2))
    net.initialize(init.Normal(sigma=0.01))

    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    loss = gloss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})

    num_epochs = 10
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, trainer)


if __name__ == "__main__":
    # basic_multilayer()
    simple_multilayer()
