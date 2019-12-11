#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/26 11:51
# @Author  : ganliang
# @File    : test_tf_kmean.py
# @Desc    : kmeans测试

import unittest
from mtensorflow.biancheng import tf_kmean


class TestTensorflowKMean(unittest.TestCase):

    def test_kmean_iris(self):
        tf_kmean.kmean_iris_show()

    def test_kmean_iris_cluster(self):
        tf_kmean.kmean_iris_cluster()

    def test_kmean_iris_distances(self):
        tf_kmean.kmean_iris_distances()


if __name__ == '__main__':
    unittest.main()