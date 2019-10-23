#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/23 21:03
# @Author  : ganliang
# @File    : test_tf_activation.py
# @Desc    : 激活函数测试类

import unittest

from src.mtensorflow import tf_activation


class TestTensorflowActivation(unittest.TestCase):

    def test_activation_threhold(self):
        tf_activation.activation_threhold()

    def test_activation_sigmoid(self):
        tf_activation.activation_sigmoid()

    def test_activation_tanh(self):
        tf_activation.activation_tanh()

    def test_activation_liner(self):
        tf_activation.activation_liner()

    def test_activation_relu(self):
        tf_activation.activation_relu()

    def test_activation_softmax(self):
        tf_activation.activation_softmax()


if __name__ == '__main__':
    unittest.main()
