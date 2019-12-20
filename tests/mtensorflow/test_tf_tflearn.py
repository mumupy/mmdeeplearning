#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/25 13:55
# @Author  : ganliang
# @File    : test_tf_tflearn.py
# @Desc    : tflearn测试
import unittest

from mtensorflow import tf_tflearn


class TestTensorflowTfLearn(unittest.TestCase):

    def test_tflearn_cifar(self):
        tf_tflearn.tflearn_cifar()

    def test_tflearn_imdb(self):
        tf_tflearn.tflearn_imdb()


if __name__ == '__main__':
    unittest.main()
