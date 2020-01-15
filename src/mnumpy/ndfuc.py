#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/20 20:27
# @Author  : ganliang
# @File    : ndfuc.py
# @Desc    : 数学函数
import numpy as np

print(np.sin(1))
print(np.sin(0.8))

a = np.array([0, 30, 45, 60, 90])
print (np.sin(a * np.pi / 180))
print (np.cos(a * np.pi / 180))
print (np.tan(a * np.pi / 180))

# a = np.array([0, 30, 45, 60, 90])
# print (np.arcsin(a * np.pi / 180))
# print (np.arccos(a * np.pi / 180))
# print (np.arctan(a * np.pi / 180))

a = np.array([1.0, 5.55, 123, 0.567, 25.532])
print  ('原数组：')
print (a)
print ('\n')
print ('舍入后：')
print (np.around(a))
print (np.around(a, decimals=1))
print (np.around(a, decimals=-1))

print ('floor后：')
print (np.floor(a))

print ('ceil后：')
print (np.ceil(a))
