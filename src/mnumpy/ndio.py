#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/21 15:10
# @Author  : ganliang
# @File    : ndio.py
# @Desc    : Numpy 可以读写磁盘上的文本数据或二进制数据。

import numpy as np

print ("save")
a = np.arange(12).reshape(3, 4)
b = np.random.rand(3, 2)
np.save("outfile.npy", a)

np.savez("outfile.npz", a, b)

print ("load")
c = np.load("outfile.npy")
print (b)

d = np.load("outfile.npz")
print (d)

print ("savetxt")
np.savetxt("outfile.txt", a)
print (np.loadtxt("outfile.txt"))
