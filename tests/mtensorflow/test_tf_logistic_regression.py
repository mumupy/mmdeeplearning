#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/23 16:53
# @Author  : ganliang
# @File    : test_tf_logistic_regression.py
# @Desc    : 逻辑回归测试
import unittest
from src.mtensorflow import tf_logistic_regression


class TestTensorflowConstant(unittest.TestCase):

    def test_logistic_regression(self):
        tf_logistic_regression.logistic_regression()


if __name__ == '__main__':
    unittest.main()
