#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/25 11:19
# @Author  : ganliang
# @File    : test_tf_cnn.py
# @Desc    : TODO
import unittest

from src.mtensorflow import tf_cnn


class TestTensorflowCnn(unittest.TestCase):

    def test_tf_cnn(self):
        tf_cnn.mnist_conv2d(epochs=10, learning_rate=0.001, batch_size=2000, dropout=0.85)


if __name__ == '__main__':
    unittest.main()
