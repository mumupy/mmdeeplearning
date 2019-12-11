#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/22 9:32
# @Author  : ganliang
# @File    : test_tf_matrix.py
# @Desc    : 矩阵测试
import unittest
from mtensorflow.biancheng import tf_operator


class TestTensorflowOperator(unittest.TestCase):

    def test_matrix_operator(self):
        tf_operator.matrix_operator()

    def test_matrix_power(self):
        tf_operator.matrix_power()

    def test_matrix_abs(self):
        tf_operator.matrix_abs()

    def test_matrix_sin(self):
        tf_operator.matrix_sin()

    def test_matrix_other(self):
        tf_operator.matrix_other()


if __name__ == '__main__':
    unittest.main()
