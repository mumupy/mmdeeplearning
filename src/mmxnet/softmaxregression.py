#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/23 19:20
# @Author  : ganliang
# @File    : softmaxregression.py
# @Desc    : softmax线性回归
import sys

import d2lzh as d2l
from mxnet.gluon import data as gdata

from src.config import logger


def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_fashion_mnist(images, labels):
    d2l.use_svg_display()
    # 这里的_表示我们忽略（不使用）的变量
    _, figs = d2l.plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.reshape((28, 28)).asnumpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    d2l.plt.show()


def mnist_train():
    mnist_train = gdata.vision.FashionMNIST(train=True)
    mnist_test = gdata.vision.FashionMNIST(train=False)
    logger.info(len(mnist_train))
    logger.info(len(mnist_test))

    X, y = mnist_train[0:9]
    logger.info(X)
    logger.info(y)
    show_fashion_mnist(X, get_fashion_mnist_labels(y))


def mnist_batch():
    """
    数据读取经常是训练的性能瓶颈，特别当模型较简单或者计算硬件性能较高时。Gluon的DataLoader中一个很方便的功能是允许使用多进程来加速数据读取
    （暂不支持Windows操作系统）。这里我们通过参数num_workers来设置4个进程读取数据
    :return:
    """
    batch_size = 256
    transformer = gdata.vision.transforms.ToTensor()
    if sys.platform.startswith('win'):
        num_workers = 0  # 0表示不用额外的进程来加速读取数据
    else:
        num_workers = 4

    mnist_train = gdata.vision.FashionMNIST(train=True)
    mnist_test = gdata.vision.FashionMNIST(train=False)

    train_iter = gdata.DataLoader(mnist_train.transform_first(transformer),
                                  batch_size, shuffle=True,
                                  num_workers=num_workers)
    test_iter = gdata.DataLoader(mnist_test.transform_first(transformer),
                                 batch_size, shuffle=False,
                                 num_workers=num_workers)
    logger.info(len(train_iter))
    logger.info(len(test_iter))


if __name__ == "__main__":
    # mnist_train()
    mnist_batch()
