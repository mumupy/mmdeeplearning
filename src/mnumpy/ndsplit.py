#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/20 18:26
# @Author  : ganliang
# @File    : ndsplit.py
# @Desc    : 数组切割
import numpy as np

a = np.arange(16).reshape(4, 4)
print(a)

print(np.split(a, 2))
print(np.hsplit(a, 4))
print(np.vsplit(a, 4))
