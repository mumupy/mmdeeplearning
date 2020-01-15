#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/20 18:42
# @Author  : ganliang
# @File    : ndbitwise.py
# @Desc    : 位运算
import numpy as  np

print(bin(123))
print(bin(np.bitwise_and(13, 18)))
print(np.bitwise_or(13, 18))
print(np.bitwise_xor(13, 18))
print(np.bitwise_not(np.arange(10)))
print(np.bitwise_not([10]))
print(np.right_shift(10, 1))
print(np.binary_repr(10, 8))
print(np.binary_repr(40, 8))
print(np.left_shift(10, 1))
