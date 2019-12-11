#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/23 22:15
# @Author  : ganliang
# @File    : test_tf_perceptron.py
# @Desc    : 感知机测试

import unittest

from mtensorflow.biancheng import tf_perceptron


class TestTensorflowConstant(unittest.TestCase):

    def test_perceptron(self):
        tf_perceptron.perceptron()

    def test_multiple_perceptron_mnist(self):
        tf_perceptron.multiple_perceptron_mnist(epochs=10, learning_rate=0.001, batch_size=200)

    def test_multiple_perceptron_boston(self):
        tf_perceptron.multiple_perceptron_boston(epochs=100, learning_rate=0.001, batch_size=100)

if __name__ == '__main__':
    unittest.main()
