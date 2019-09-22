#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/22 22:08
# @Author  : ganliang
# @File    : mndarray.py
# @Desc    : mndarry包测试
import unittest

from src.mmxnet import mndarray


class TestNdarray(unittest.TestCase):
    """
    测试mxnet包下的ndarray模块
    """

    def test_arange(self):
        mndarray.arange()

    def test_reshape(self):
        mndarray.reshape()

    def test_ones(self):
        mndarray.ones()

    def test_zeros(self):
        mndarray.zeros()

    def test_random(self):
        mndarray.random()


if __name__ == '__main__':
    unittest.main()
