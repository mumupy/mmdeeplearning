#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/22 16:36
# @Author  : ganliang
# @File    : test_tf_constant.py
# @Desc    : tensorflow常量测试
import unittest
from src.mtensorflow import tf_constant


class TestTensorflowConstant(unittest.TestCase):

    def test_constant_helloworld(self):
        tf_constant.helloworld()

    def test_constant_add(self):
        tf_constant.add()

    def test_constant_interactive_add(self):
        tf_constant.interactive_add()

    def test_constant_constant(self):
        tf_constant.constant()

    def test_constant_zero_onse(self):
        tf_constant.zero_onse()

    def test_constant_linspace(self):
        tf_constant.linspace()

    def test_constant_range(self):
        tf_constant.range()

    def test_constant_random(self):
        tf_constant.random()



if __name__ == '__main__':
    unittest.main()