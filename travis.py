#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/22 18:18
# @Author  : ganliang
# @File    : travis.py
# @Desc    : travis单元测试

import unittest


class TestMethods(unittest.TestCase):

    def test_add(self):
        self.assertEqual(":)", ":)")


if __name__ == '__main__':
    unittest.main()
