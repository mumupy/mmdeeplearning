#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/22 22:34
# @Author  : ganliang
# @File    : runner.py
# @Desc    : 测试所有的测试用例

import unittest

import xmlrunner


def test_all():
    """
    测试测试模块下所有的测试脚本
    :return:
    """
    suite = unittest.TestSuite()
    all_cases = unittest.defaultTestLoader.discover('.', 'test_*.py')
    for case in all_cases:
        suite.addTests(case)

    runner = xmlrunner.XMLTestRunner(output='report')
    runner.run(suite)


if __name__ == "__main__":
    test_all()
