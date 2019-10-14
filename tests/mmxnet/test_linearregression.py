#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/23 19:54
# @Author  : ganliang
# @File    : test_linearregression.py
# @Desc    : 线性回归
import unittest

from src.mmxnet import linearregression


class TestLinearRegression(unittest.TestCase):
    """
    测试mxnet包下的linearregression模块
    """

    def test_houseprise(self):
        linearregression.house_prise()

    def test_linereg(self):
        linearregression.linereg_plg()

    def test_linearregression(self):
        linearregression.linearregression()

    def test_linergluon(self):
        linearregression.liner_gluon()

    def test_houseprisegulon(self):
        linearregression.house_prise_gulon()


if __name__ == '__main__':
    unittest.main()
