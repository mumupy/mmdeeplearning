#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/27 13:21
# @Author  : ganliang
# @File    : __init__.py.py
# @Desc    : keras高級api

from matplotlib import pyplot as plt


def keras_history_plotcurve(history):
    """
    绘制损失函数
    :param history:
    :return:
    """
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
