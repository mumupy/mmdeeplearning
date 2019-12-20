#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/26 20:49
# @Author  : ganliang
# @File    : test_tf_gym.py
# @Desc    : 强化学习测试
import unittest
from mtensorflow import tf_gym


class TestTensorflowConstant(unittest.TestCase):

    def test_gym_basic(self):
        tf_gym.gym_basic()


if __name__ == '__main__':
    unittest.main()