#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/21 11:15
# @Author  : ganliang
# @File    : ndsort.py
# @Desc    : 排序
import numpy as np

a = np.array([[3, 7], [9, 1]])
print ('我们的数组是：')
print (a)
print ('\n')
print ('调用 sort() 函数：')
print (np.sort(a))
print ('\n')
print ('按列排序：')
print (np.sort(a, axis=0))
print ('\n')
# 在 sort 函数中排序字段
dt = np.dtype([('name', 'S10'), ('age', int)])
a = np.array([("raju", 21), ("anil", 25), ("ravi", 17), ("amar", 27)], dtype=dt)
print ('我们的数组是：')
print (a)
print ('\n')
print ('按 name 排序：')
print (np.sort(a, order='name'))

x = np.array([3, 1, 2])
print ('我们的数组是：')
print (x)
print ('\n')
print ('对 x 调用 argsort() 函数：')
y = np.argsort(x)
print (y)
print ('\n')
print ('以排序后的顺序重构原数组：')
print (x[y])
print ('\n')
print ('使用循环重构原数组：')
for i in y:
    print (x[i])

nm = ('raju', 'anil', 'ravi', 'amar')
dv = ('f.y.', 's.y.', 's.y.', 'f.y.')
# pv = ('1', '2', '3', '4')
ind = np.lexsort((dv, nm))
print ('调用 lexsort() 函数：')
print (ind)
print ('\n')
print ('使用这个索引来获取排序后的数据：')
print ([nm[i] + ", " + dv[i] for i in ind])

a = np.array([1, 256, 8755], dtype=np.int16)
print ('我们的数组是：')
print (a)
print ('以十六进制表示内存中的数据：')
print (map(hex, a))
# byteswap() 函数通过传入 true 来原地交换
print ('调用 byteswap() 函数：')
print (a.byteswap(True))
print ('十六进制形式：')
print (map(hex, a))
# 我们可以看到字节已经交换了
