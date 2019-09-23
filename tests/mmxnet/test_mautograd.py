#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/23 19:59
# @Author  : ganliang
# @File    : test_mautograd.py
# @Desc    : 测试梯度算法实例
import unittest

from src.mmxnet import mautograd


class TestLinearRegression(unittest.TestCase):
    """
    测试mxnet包下的autograd模块
    """

    def test_autograd(self):
        mautograd.auto_grad()

    def test_controlgrad(self):
        mautograd.control_grad()


if __name__ == '__main__':
    unittest.main()
