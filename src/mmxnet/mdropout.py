#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/26 18:12
# @Author  : ganliang
# @File    : mdropout.py
# @Desc    : 过拟合解决办法-丢弃发
import d2lzh as d2l
from mxnet import nd, gluon, autograd, init
from mxnet.gluon import nn, loss as gloss

from src.config import logger


def dropout(X, drop_rate):
    """
    丢弃X数据集中的rate比例数据
    :param X:  数据集合
    :param drop_rate:  丢弃比例
    :return:
    """
    assert 0 <= drop_rate <= 1
    logger.info("丢弃之前的数据:\n%s" % X)

    keep_rate = 1 - drop_rate
    if keep_rate == 0:
        return nd.zeros_like(X)

    drop_x = nd.random.uniform(0, 1, shape=X.shape)
    logger.info("丢弃随机数据:%s" % drop_x)

    mask = drop_x < keep_rate

    logger.info("丢弃之后数据:\n%s" % mask)
    return mask * X


def dropout2(X, drop_rate):
    autograd.set_training(True)
    Z = nd.zeros_like(X)
    nd.Dropout(X, p=drop_rate, out=Z)
    return Z


def dropout_gluon():
    drop_prob1, drop_prob2, lr, batch_size, num_epochs = 0.2, 0.5, 0.1, 64, 50

    net = nn.Sequential()
    net.add(nn.Dense(256, activation="relu"),
            nn.Dropout(drop_prob1),  # 在第一个全连接层后添加丢弃层
            nn.Dense(256, activation="relu"),
            nn.Dropout(drop_prob2),  # 在第二个全连接层后添加丢弃层
            nn.Dense(10))
    net.initialize(init.Normal(sigma=0.01))

    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    loss = gloss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None,
                  None, trainer)


if __name__ == "__main__":
    x = nd.arange(64).reshape(8, 8)
    logger.info(dropout(x, 0.5))
    logger.info(dropout2(x, 0.5) / 2)

    dropout_gluon()
