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
    logger.info("ndarray a:")
    logger.info(a)

    logger.info("ndarray nd.arange(10, 20, 2, dtype=np.float32, repeat=2:")
    logger.info(nd.arange(10, 20, 2, dtype=np.float32, repeat=2))

    logger.info("ndarray a.shape:")
    logger.info(a.shape)

    logger.info("ndarray a.size:")
    logger.info(a.size)

    logger.info("ndarray a.T:")
    logger.info(a.T)


def matrix():
    """
    j矩阵 设A为 m*p 的矩阵，B为 p*n 的矩阵，那么称 m*n 的矩阵C为乘积，C=AB
    :return:
    """
    a = nd.arange(12).reshape(3, 4)
    print("ndarray a:")
    logger.info(a)

    b = nd.arange(8).reshape(4, 2)
    print("ndarray b:")
    logger.info(b)

    logger.info("ndarray nd.dot(a, b):")
    logger.info(nd.dot(a, b))


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


def normal():
    """
    它的每个元素都随机采样于均值为0、标准差为1的正态分布。nd.sqrt(nd.power(a, 2).sum())
    :return:
    """
    n = nd.normal(0, 1, shape=(2, 2))
    logger.info(n)

    a = nd.array([1, 2, 3, 4])
    print(a.norm())
    print(nd.sqrt(nd.power(a, 2).sum()))


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


def operator():
    """
    mxnet的ndarray加减乘除
    :return:
    """
    a = nd.arange(12).reshape(3, 4)
    b = nd.uniform(0, 10, shape=(3, 4))
    logger.info("ndarray a:")
    logger.info(a)
    logger.info("ndarray b:")
    logger.info(b)

    logger.info("ndarray a.exp():")
    logger.info(a.exp())

    logger.info("ndarray a.sum():")
    logger.info(a.sum())

    logger.info("ndarray a.max():")
    logger.info(a.max())

    logger.info("ndarray a.min():")
    logger.info(a.min())

    logger.info("ndarray a.abs():")
    logger.info(a.abs())

    logger.info("ndarray a.norm().asscalar():")
    logger.info(a.norm().asscalar())

    nd_add = a + b
    logger.info("ndarray a+b:")
    logger.info(nd_add)

    nd_sub = a - b
    logger.info("ndarray a-b:")
    logger.info(nd_sub)

    nd_mul = a * b
    logger.info("ndarray a*b:")
    logger.info(nd_mul)

    nd_dev = a / b
    logger.info("ndarray a/b:")
    logger.info(nd_dev)

    nd_dot = nd.dot(a, a.T)
    logger.info("ndarray nd.dot(a,a.T):")
    logger.info(nd_dot)

    # logger.info("ndarray nd.batch_dot(a,a.T):")
    # logger.info(
    #     nd.batch_dot(nd.arange(24).reshape(2, 3, 4), nd.arange(24).reshape(2, 3, 4), nd.arange(24).reshape(2, 3, 4)))

    logger.info("ndarray nd.concat(a,b,dim=0)")
    logger.info(nd.concat(a, b, dim=0))
    logger.info("ndarray nd.concat(a,b,dim=1)")
    logger.info(nd.concat(a, b, dim=1))


def broadcasting():
    """
    前面我们看到如何对两个形状相同的NDArray做按元素运算。当对两个形状不同的NDArray按元素运算时，可能会触发广播（broadcasting）机制：
    先适当复制元素使这两个NDArray形状相同后再按元素运算。
    :return:
    """
    a = nd.arange(12).reshape(3, 4)
    b = nd.array([[1, 2, 3, 4]])
    logger.info("ndarray a:")
    logger.info(a)
    logger.info("ndarray b:")
    logger.info(b)

    logger.info("ndarray a+b")
    logger.info(a + b)

    logger.info("ndarray a*b")
    logger.info(a * b)


def indexing():
    """
    NDArray中，索引（index）代表了元素的位置。NDArray的索引从0开始逐一递增。例如，一个3行2列的矩阵的行索引分别为0、1和2，列索引分别为0和1。
    :return:
    """
    a = nd.arange(12).reshape(3, 4)
    logger.info("ndarray a:")
    logger.info(a)

    logger.info("ndarray a[1, 2]")
    logger.info(a[1, 2])

    logger.info("ndarray a[1, 2]")
    logger.info(a[1:, 2:])


def ndarray_numpy_transform():
    """
    ndarray 和numpy数据类型互换互换
    :return:
    """
    a = nd.arange(12).reshape(3, 4)
    logger.info("ndarray a:")
    logger.info(a)

    n = a.asnumpy()
    logger.info("ndarray a.asnumpy():")
    logger.info(n)
    logger.info("ndarray type(n):")
    logger.info(type(n))


if __name__ == "__main__":
    # arange()
    matrix()
    # reshape()
    # ones()
    # zeros()
    # random()
    # normal()
    # operator()
    # broadcasting()
    # indexing()
    # ndarray_numpy_transform()
