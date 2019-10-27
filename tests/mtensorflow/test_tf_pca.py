#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/26 10:45
# @Author  : ganliang
# @File    : test_tf_pca.py
# @Desc    : 主成分分析

import unittest

from src.mtensorflow import tf_pca


class TestTensorflowPca(unittest.TestCase):

    def test_pca_basic(self):
        tf_pca.pca_basic()

    def test_pca_embeddings(self):
        tf_pca.pca_embeddings()

if __name__ == '__main__':
    unittest.main()
