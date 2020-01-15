#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/21 14:42
# @Author  : ganliang
# @File    : linalg.py
# @Desc    : 线性代数

import numpy as np

a = np.array([[1, 2], [3, 4]])
b = np.array([[11, 12], [13, 14]])

print ("dot函数在是二维矩阵的时候就是矩阵相乘")
print (np.matrix(a) * np.matrix(b))
print ("dot下标元素的乘积和")
print(np.dot(a, b))

print ("\nvdot两个向量的点积")
# vdot 将数组展开计算内积 两个向量的点积。 如果第一个参数是复数，那么它的共轭复数会用于计算。 如果参数是多维数组，它会被展开。
print (np.vdot(a, b))

print ("\ninner向量内积")
print (np.inner(np.array([1, 2, 3]), np.array([0, 1, 0])))
# 等价于 1*0+2*1+3*0

print (np.inner(np.array([[1, 2], [3, 4]]), np.array([[11, 12], [13, 14]])))

print ("\nmatmul矩阵乘积")
a = [[1, 0], [0, 1]]
b = [[4, 1], [2, 2]]
print (np.matmul(a, b))

print ("\nlinalg.det 计算输入矩阵的行列式")
a = np.array([[1, 2], [3, 4]])
print (np.linalg.det(a))

print ("\nlinalg.solve 矩阵形式的线性方程的解")
a = np.array([[3, 1], [1, 2]])
b = np.array([9, 8])
x = np.linalg.solve(a, b)
print (x)

print ("\nlinalg.solve 乘法逆矩阵")
x = np.array([[1, 2], [3, 4]])
y = np.linalg.inv(x)
print (x)
print (y)
print (np.dot(x, y))

a = np.array([[1, 1, 1], [0, 2, 5], [2, 5, -1]])
b = np.array([[6], [-4], [27]])
x = np.linalg.solve(a, b)
print (x)
