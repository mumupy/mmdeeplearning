#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/22 16:30
# @Author  : ganliang
# @File    : msymbol.py
# @Desc    : 网络
from mxnet import symbol

s = symbol.random_normal(0, 1, (3, 4))
print(s)