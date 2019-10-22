#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/22 11:26
# @Author  : ganliang
# @File    : test_tf_matrix.py
# @Desc    : 矩阵测试

import unittest
from src.mtensorflow import tf_matrix


class TestTensorflowMatrix(unittest.TestCase):

    def test_matrix_matmul(self):
        tf_matrix.matrix_matmul()

    def test_matrix_transpose(self):
        tf_matrix.matrix_transpose()

    def test_matrix_trace(self):
        tf_matrix.matrix_trace()

    def test_matrix_determinant(self):
        tf_matrix.matrix_determinant()

    def test_matrix_inverse(self):
        tf_matrix.matrix_inverse()

    def test_matrix_svd(self):
        tf_matrix.matrix_svd()

    def test_matrix_qr(self):
        tf_matrix.matrix_qr()

    def test_matrix_diag(self):
        tf_matrix.matrix_diag()

    def test_matrix_reduce(self):
        tf_matrix.matrix_reduce()

    def test_matrix_cumsum(self):
        tf_matrix.matrix_cumsum()

    def test_matrix_segment(self):
        tf_matrix.matrix_segment()

    def test_matrix_condition(self):
        tf_matrix.matrix_condition()


if __name__ == '__main__':
    unittest.main()
