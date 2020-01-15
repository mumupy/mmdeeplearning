#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/21 9:55
# @Author  : ganliang
# @File    : arithmetic.py
# @Desc    : 算数运算
import numpy as np

a = np.arange(9, dtype=np.float_).reshape(3, 3)
print ('\n第一个数组：')
print (a)
print ('\n第二个数组：')
b = np.array([10, 10, 10])
print (b)
print ('\n两个数组相加：')
print (np.add(a, b))
print ('\n两个数组相减：')
print (np.subtract(a, b))
print ('\n两个数组相乘：')
print (np.multiply(a, b))
print ('\n两个数组相除：')
print (np.divide(a, b))

print ('\nreciprocal')
a = np.arange(10, 20, 2, dtype=np.float)
print (a)
print (np.reciprocal(a))

print (np.reciprocal(np.array([0.03, 12, 23])))

print ('\npower：')
print (np.power(np.array([0.3, 10, 30]), 1))

print ('\nmod：')
print (np.mod(np.array([13, 5, 30]), 3))

print ('\nremainder：')
print (np.remainder(np.array([13, 5, 30]), 3))
