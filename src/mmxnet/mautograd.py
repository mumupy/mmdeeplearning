#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/23 10:27
# @Author  : ganliang
# @File    : mautograd.py
# @Desc    : 自动求梯度
from mxnet import autograd, nd

from src.config import logger


def cal_grad(X):
    """
    计算梯度
    :return:
    """
    logger.info(X)
    logger.info(X.T)
    X.attach_grad()
    with autograd.record():
        y = 2 * X
        logger.info(y)
    y.backward()
    logger.info(X.grad)


def auto_grad():
    """
    对函数  y=2x**2  求关于列向量  x  的梯度 4x
    :return:
    """
    x = nd.arange(4).reshape((4, 1))
    logger.info("autograd 数组:")
    logger.info(x)

    # 调用attach_grad函数来申请存储梯度所需要的内存
    x.attach_grad()

    logger.info("autograd.is_training():")
    logger.info(autograd.is_training())
    # 调用record函数来要求MXNet记录与求梯度有关的计算。
    with autograd.record():
        y = 2 * nd.dot(x.T, x)
        logger.info(autograd.is_training())
        logger.info(y)

    # 调用backward函数自动求梯度
    y.backward()

    logger.info("autograd 梯度:")
    logger.info(x.grad)


def control_grad():
    """
    即使函数的计算图包含了Python的控制流（如条件和循环控制）
    :return:
    """

    def f(a):
        """
        f(a) = x * a
        :param a:
        :return:
        """
        b = a * 2
        while b.norm().asscalar() < 1000:
            b = b * 2
        if b.sum().asscalar() > 0:
            c = b
        else:
            c = 100 * b
        return c

    a = nd.random.normal(shape=1)
    logger.info("autograd a:")
    logger.info(a)

    a.attach_grad()
    with autograd.record():
        c = f(a)
        logger.info(c)
    c.backward()

    logger.info("autograd a.grad:")
    logger.info(a.grad)


if __name__ == "__main__":
    # cal_grad(nd.arange(6).reshape(6, 1))
    # auto_grad()
    control_grad()
