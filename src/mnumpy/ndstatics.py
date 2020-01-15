#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/21 10:24
# @Author  : ganliang
# @File    : ndstatics.py
# @Desc    : 统计学函数

# 如果将三维数组的每一个二维看做一个平面（plane，X[0, :, :], X[1, :, :], X[2, :, :]），
# 三维数组即是这些二维平面层叠（stacked）出来的结果。则（axis=0）表示全部平面上的对应位置，（axis=1），
# 每一个平面的每一列，（axis=2），每一个平面的每一行。

import numpy as np

a = np.array([[3, 7, 5], [8, 4, 3], [2, 4, 9]])
print(a.shape)
print('我们的数组是：')
print(a)
print('\n')
print('沿轴 1 调用 amin() 函数：')
print(np.amin(a, 1))
print('\n')
print('沿轴 0 再次调用 amin() 函数：')
print(np.amin(a, 0))
print('\n')
print('沿轴 1 调用 amax() 函数：')
print(np.amax(a, axis=1))
print('\n')
print('沿轴 0 再次调用 amax() 函数：')
print(np.amax(a, axis=0))

print(np.ptp(a))
print('\n')
print('沿轴 1 调用 ptp() 函数：')
print(np.ptp(a, axis=1))
print('\n')
print('沿轴 0 调用 ptp() 函数：')
print(np.ptp(a, axis=0))

print('调用 percentile() 函数：')
# 50% 的分位数，就是 a 里排序之后的中位数
print(np.percentile(a, 50))

# axis 为 0，在纵列上求
print(np.percentile(a, 50, axis=0))

# axis 为 1，在横行上求
print(np.percentile(a, 50, axis=1))

# 保持维度不变
print(np.percentile(a, 50, axis=1, keepdims=True))

print('调用 median() 函数：')
print(np.median(a))
print('\n')
print('沿轴 0 调用 median() 函数：')
print(np.median(a, axis=0))
print('\n')
print('沿轴 1 调用 median() 函数：')
print(np.median(a, axis=1))

# mean算术平均值是沿轴的元素的总和除以元素的数量。
print('调用 mean() 函数：')
print(np.mean(a))
print('\n')
print('沿轴 0 调用 mean() 函数：')
print(np.mean(a, axis=0))
print('\n')
print('沿轴 1 调用 mean() 函数：')
print(np.mean(a, axis=1))

# average平均值，可以附带权重
print('调用 average() 函数：')
print(np.average(a))
print('\n')
# 不指定权重时相当于 mean 函数
print('再次调用 average() 函数：')
print(np.average(a, weights=np.array([[4, 3, 1], [2, 1, 3], [4, 1, 2]])))

# 标注差 std = sqrt(mean((x - x.mean())**2))
print('\n标准差 std() 函数：')
print(np.std(a))

# 方差统计中的方差（样本方差）是每个样本值与全体样本值的平均数之差的平方值的平均数，即 mean((x - x.mean())** 2)。换句话说，标准差是方差的平方根。
print('\n方差 var() 函数：')
print(np.var(a))
