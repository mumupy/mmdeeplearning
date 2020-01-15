#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/21 14:23
# @Author  : ganliang
# @File    : npmatlib.py
# @Desc    : 矩阵
import numpy as np
import numpy.matlib as ml

print ("empty")
print(ml.empty((3, 3), dtype=np.int, order='F'))
print(ml.empty((3, 3), dtype=np.int, order='C'))

print ("\nzeros")
print(ml.zeros((3, 3), dtype=np.int, order='C'))

print ("\nones")
print(ml.ones((3, 3), dtype=np.int, order='C'))

print ("\neye")
print(ml.eye(3, dtype=np.int))

print ("\nidentity")
print(ml.identity(3, dtype=np.int))

print ("\nrand")
print(ml.rand(2, 3))

print ("\nmatrix")
a = np.arange(12).reshape(3, 4)
mr = ml.matrix(a)
print (a)
print (mr)
print (type(a))
print (type(mr))
