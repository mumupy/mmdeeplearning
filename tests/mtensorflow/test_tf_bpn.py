#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/24 9:37
# @Author  : ganliang
# @File    : test_tf_bpn.py
# @Desc    : bpn反向传播
import unittest
from src.mtensorflow import tf_bpn


class TestTensorflowBpn(unittest.TestCase):

    def test_bpn(self):
        tf_bpn.bpn(epochs=1, batch_size=1000, learning_rate=0.01)


if __name__ == '__main__':
    unittest.main()
