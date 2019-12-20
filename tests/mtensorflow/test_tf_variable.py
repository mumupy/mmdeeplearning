#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/21 17:36
# @Author  : ganliang
# @File    : test_tf_variable.py
# @Desc    : 测试tf_variable脚本
import unittest
from mtensorflow import tf_variable


class TestTensorflowVariable(unittest.TestCase):

    def test_variable(self):
        tf_variable.variable()

    def test_saver(self):
        tf_variable.saver()

    def test_restore(self):
        tf_variable.restore()

    def test_placeholder(self):
        tf_variable.placeholder()

    def test_convert_to_tensor(self):
        tf_variable.convert_to_tensor()

    def test_range(self):
        tf_variable.range()


if __name__ == '__main__':
    unittest.main()
