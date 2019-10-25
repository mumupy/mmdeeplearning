#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/24 18:30
# @Author  : ganliang
# @File    : test_tf_keras.py
# @Desc    : keras测试

import unittest

from src.mtensorflow import tf_keras


class TestTensorflowKeras(unittest.TestCase):

    def test_keras_boston(self):
        tf_keras.keras_boston(epochs=100, learning_rate=0.001, batch_size=100)

    def test_super_variables(self):
        tf_keras.super_variables(epochs=2, min_loss=0.04, batch_size=100)

    def test_image_classify(self):
        tf_keras.image_classify()


if __name__ == '__main__':
    unittest.main()
