#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/19 22:50
# @Author  : ganliang
# @File    : ndarray.py
# @Desc    : numpy数组

import numpy as np

print(np.array([1, 2, 3], dtype=int))
print(np.array([[[1, 2, 3], [1, 3, 2], [3, 4, 2]], [[1, 3, 2], [1, 3, 2], [3, 5, 2]]], dtype=complex, ndmin=3))

# int8, int16, int32, int64 四种数据类型可以使用字符串 'i1', 'i2','i4','i8' 代替
print(np.dtype("int32"))

print(np.dtype("int32"))
print(np.dtype([("qianqian", np.int8)]))

student = np.dtype([('name', 'S20'), ('age', 'i1'), ('marks', 'f4')])
print(student)

print(np.array([], dtype=student, ndmin=2))
a = np.array([('abc', 21, 50), ('xyz', 18, 75)], dtype=student)
print(a["name"])
print(a[0])

a = np.arange(24, dtype=np.int64)
print(a)

# 维度 行数  列数
shape = a.reshape((4, 2, 3))
print(shape)
print(shape.shape)
print(shape.itemsize)
print(shape.flags)

a = np.empty([1, 2, 3, 3], dtype=np.int64)
print(a)

x = np.zeros(5)
print(x)

y = np.zeros((5,), dtype=np.int)
print(y)

z = np.zeros((3, 3, 4), dtype=[('x', 'i4'), ('y', 'i4')])
print(z)

x = np.ones([2, 3], dtype=int)
print(x)

x = [1, 2, 3]
a = np.asarray(x)
print(a)
print(np.array(x))

s = b'Hello World'
a = np.frombuffer(s, dtype="S1")
print(a)

it = iter(range(5))
# 使用迭代器创建 ndarray
x = np.fromiter(it, dtype=np.int64)
print(x)

a = np.linspace(1, 100, 10)
print(a)

a = np.logspace(1.0, 2.0, num=10, base=100, endpoint=False)
print(a)

ar = np.arange(1, 10, 2, dtype=np.int64)
print(ar)
print(np.arange(10))

a = np.array([[1, 2, 3], [3, 4, 5], [4, 5, 6]])
print(a[..., 1])
print(a[1])
print(a[1, ...])
print(a[..., 1:])

x = np.array([[1, 2], [3, 4], [5, 6]])
print(x)
y = x[[0, 1, 2], [0, 1, 0]]  # 00 11 20
print(y)

x = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
rows = np.array([[0, 0], [3, 3]])
cols = np.array([[0, 2], [0, 2]])
y = x[rows, cols]
print(y)

x = np.arange(32).reshape((8, 4))
print(x)
print(x[[4, 2, 1, 7]])
print(x[[-4, -2, -1, -7]])

b = np.array([1, 2, 3])
# 将一个数组 拓展到多少被 4行2倍
bb = np.tile(b, (5, 2))
print(bb)

a = np.arange(6).reshape(2, 3)
print(a.T)
print(np.transpose(a))
print(a)
for x in np.nditer(a):
    print(x)

a = np.array([[1, 2, 3], [3, 4, 5]])

print('第一个数组：')
print(a)
print('\n')
b = np.array([[5, 6, 7], [7, 8, 9]])

print('第二个数组：')
print(b)
print('\n')
# 两个数组的维度相同

print('沿轴 0 连接两个数组：')
print(np.concatenate((a, b)))
print('\n')

print('沿轴 1 连接两个数组：')
print(np.concatenate((a, b), axis=1))

print('沿轴 0 堆叠两个数组：')
print(np.stack((a, b), 0))
print('\n')

print('沿轴 1 堆叠两个数组：')
print(np.stack((a, b), 1))

print('水平堆叠：')
c = np.hstack((a, b))
print(c)

print('垂直堆叠：')
c = np.vstack((a, b))
print(c)
