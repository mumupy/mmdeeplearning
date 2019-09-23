#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/23 12:53
# @Author  : ganliang
# @File    : linearregression.py
# @Desc    : 线性回归
import math
import random

from mxnet import init
from mxnet import nd, autograd, gluon
from mxnet.gluon import data as gdata
from mxnet.gluon import loss as gloss
from mxnet.gluon import nn

from src.config import logger


def house_prise():
    """
    使用线性回归估算房价
    :return:
    """

    x = nd.array([[120, 2], [100, 1], [130, 3]])

    y = nd.array([[1300000], [980000], [1400000]])

    logger.info("x:")
    logger.info(x)

    logger.info("y:")
    logger.info(y)

    w = nd.broadcast_div(y, x)
    logger.info("nd.broadcast_div(y, x):")
    logger.info(w)

    logger.info("nd.sum(w, axis=0):")
    aw = nd.sum(w, axis=0) / 3
    logger.info(aw)

    print((aw * nd.array([100, 2])).norm())


def linereg():
    from IPython import display
    from matplotlib import pyplot as plt
    from mxnet import nd

    num_inputs = 2
    num_examples = 1000
    true_w = [2, -3.4]
    true_b = 4.2
    features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
    labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
    labels += nd.random.normal(scale=0.01, shape=labels.shape)

    def use_svg_display():
        # 用矢量图显示
        display.set_matplotlib_formats('svg')

    def set_figsize(figsize=(3.5, 2.5)):
        use_svg_display()
        # 设置图的尺寸
        plt.rcParams['figure.figsize'] = figsize

    set_figsize()
    plt.scatter(features[:, 1].asnumpy(), labels.asnumpy(), 1)
    plt.show()


def linearregression():
    def linreg(X, w, b):
        """
        计算模型
        :param X:
        :param w:
        :param b:
        :return:
        """
        return nd.dot(X, w) + b

    def squared_loss(y_hat, y):
        """
        计算损失函数
        :param y_hat:
        :param y:
        :return:
        """
        return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

    def sgd(params, lr, batch_size):
        """
        优化函数
        :param params:
        :param lr:
        :param batch_size:
        :return:
        """
        for param in params:
            param[:] = param - lr * param.grad / batch_size

    def data_iter(batch_size, features, labels):
        """
        迭代取数据
        :param batch_size:
        :param features:
        :param labels:
        :return:
        """
        num_examples = len(features)
        indices = list(range(num_examples))
        random.shuffle(indices)  # 样本的读取顺序是随机的
        for i in range(0, num_examples, batch_size):
            j = nd.array(indices[i: min(i + batch_size, num_examples)])
            yield features.take(j), labels.take(j)  # take函数根据索引返回对应元素

    num_inputs = 2
    num_examples = 1000
    true_w = [2, -3.4]
    true_b = 4.2
    features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
    labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
    labels += nd.random.normal(scale=0.01, shape=labels.shape)
    logger.info(labels)

    batch_size = 10
    lr = 0.03
    num_epochs = 3
    w = nd.random.normal(scale=0.01, shape=(num_inputs, 1))
    b = nd.zeros(shape=(1,))
    logger.info("w:nd.random.normal(scale=0.01, shape=(num_inputs, 1))")
    logger.info(w)
    logger.info("b:nd.zeros(shape=(1,))")
    logger.info(b)
    w.attach_grad()
    b.attach_grad()

    for epoch in range(num_epochs):  # 训练模型一共需要num_epochs个迭代周期
        # 在每一个迭代周期中，会使用训练数据集中所有样本一次（假设样本数能够被批量大小整除）。X
        # 和y分别是小批量样本的特征和标签
        for X, y in data_iter(batch_size, features, labels):
            with autograd.record():
                l = squared_loss(linreg(X, w, b), y)  # l是有关小批量X和y的损失
            l.backward()  # 小批量的损失对模型参数求梯度
            sgd([w, b], lr, batch_size)  # 使用小批量随机梯度下降迭代模型参数
        train_l = squared_loss(linreg(features, w, b), labels)
        logger.info('epoch %d, loss %f' % (epoch + 1, train_l.mean().asnumpy()))

    logger.info("真实值:")
    logger.info(true_w)
    logger.info(true_b)

    logger.info("训练值:")
    logger.info(w)
    logger.info(b)


def liner_gluon():
    """
    线性回归 gluon实现
    :return:
    """

    num_inputs = 2
    num_examples = 1000
    true_w = [2, -3.4]
    true_b = 4.2
    features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
    labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
    labels += nd.random.normal(scale=0.01, shape=labels.shape)

    logger.info(features)
    logger.info(labels)
    batch_size = 10
    # 将训练数据的特征和标签组合
    dataset = gdata.ArrayDataset(features, labels)
    # 随机读取小批量
    data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)

    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(init.Normal(sigma=0.01))
    loss = gloss.L2Loss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})

    num_epochs = 3
    for epoch in range(1, num_epochs + 1):
        for X, y in data_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        l = loss(net(features), labels)
        logger.info('epoch %d, loss: %f' % (epoch, l.mean().asnumpy()))

    dense = net[0]
    logger.info("真实数据")
    logger.info(true_w)
    logger.info(true_b)
    logger.info("预测数据")
    logger.info(dense.weight.data())
    logger.info(dense.bias.data())


def house_prise_gulon():
    """
    使用gulon模型构建房价预估
    :return:
    """
    features = nd.array(nd.array([[120, 2], [100, 1], [130, 3]]))
    labels = nd.array([1200000, 1000000, 1300000])
    logger.info(features)
    logger.info(labels)
    # labels += nd.random.normal(scale=0.01, shape=labels.shape)

    batch_size = 10
    # 将训练数据的特征和标签组合
    dataset = gdata.ArrayDataset(features, labels)
    # 随机读取小批量
    data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(init.Normal(sigma=0.01))
    loss = gloss.L2Loss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})

    num_epochs = 3
    for epoch in range(1, num_epochs + 1):
        for X, y in data_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        l = loss(net(features), labels)
        logger.info('epoch %d, loss: %f' % (epoch, l.mean().asnumpy()))

    dense = net[0]
    logger.info("预测数据")
    logger.info(dense.weight.data())
    logger.info(dense.bias.data())


if __name__ == "__main__":
    # house_prise()
    # linereg()
    # linearregression()
    # liner_gluon()
    # house_prise_gulon()
    logger.info(math.log(0.1))
    logger.info(math.log(8, 2))
    logger.info(math.log(0.2, 2))
