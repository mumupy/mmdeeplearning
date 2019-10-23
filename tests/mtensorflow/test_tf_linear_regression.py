#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/23 15:21
# @Author  : ganliang
# @File    : test_tf_linear_regression.py
# @Desc    : 测试线性回归
import unittest
from src.mtensorflow import tf_linear_regression


class TestTensorflowConstant(unittest.TestCase):

    def test_linear_regression(self):
        tf_linear_regression.linear_regression()

    def test_k_linear_regression(self):
        tf_linear_regression.k_linear_regression()

    def test_multiple_linear_regression(self):
        tf_linear_regression.multiple_linear_regression()


if __name__ == '__main__':
    unittest.main()
