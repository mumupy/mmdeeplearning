#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/22 14:55
# @Author  : ganliang
# @File    : mndarray.py
# @Desc    : mxnet的数据模型 ndarray
import numpy as np
from mxnet import nd

from src.config import logger


def arange():
    """
    创建连续的数组 可以指定开始 结束 步长
    :return:
    """
    a = nd.arange(10, dtype=np.float32)
    logger.info(a)
    logger.info(a.shape)

    a = nd.arange(10, 20, 2, dtype=np.float32, repeat=2)
    logger.info(a)


def reshape():
    """
    对多维数组进行维度转换 行转列 列转行
    :return:
    """
    a = nd.arange(20).reshape(4, 5)
    logger.info(a)

    # 将20个一维数组转化为由2个二维(2行5列)组成的三维数组
    b = nd.arange(20).reshape(2, 2, 5)
    logger.info(b)


def ones():
    """
    创建一个全是1的多维数组
    :return:
    """
    a = nd.ones((3, 2), dtype=np.int32)
    logger.info(a)

    b = nd.arange(12).reshape(3, 4)
    logger.info(b)
    c = nd.ones_like(b)
    logger.info(c)


def zeros():
    """
    创建一个全是0的多维数组
    :return:
    """
    a = nd.zeros((2, 3, 2), dtype=np.int32)
    logger.info(a)

    b = nd.arange(12).reshape(3, 4)
    logger.info(b)
    c = nd.zeros_like(b)
    logger.info(c)


def random():
    """
    随机生成多维数组
    :return:
    """
    # a = nd.random_randint(low=0, high=5, shape=(2, 2, 2))
    a = nd.uniform(low=0, high=5, shape=(2, 2, 2))
    logger.info(a)

    b = nd.random_normal(0, 1, (3, 4))
    logger.info(b)

    # 从指数分布中随机抽取样本
    c = nd.random_exponential(lam=2, shape=(2, 2))
    logger.info(c)


if __name__ == "__main__":
    # arange()
    # reshape()
    # ones()
    zeros()
    # random()
