#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/21 14:06
# @Author  : ganliang
# @File    : ndcopy.py
# @Desc    : 数据复制

import numpy as np

a = np.arange(10)
b = np.copy(a)
print ("修改之前:")
print (a)
print (id(a))
print (b)
print (id(b))

a[0] = 100
print ("修改之后:")
print (a)
print (id(a))
print (b)
print (id(b))

c=a.reshape(5, 2)
print (c)