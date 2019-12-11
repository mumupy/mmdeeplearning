#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/22 10:40
# @Author  : ganliang
# @File    : __init__.py.py
# @Desc    : numpy

import matplotlib.pyplot as plt
import numpy as np

X = np.arange(0, 6, 0.1)
Y = np.cos(X)
plt.plot(X, Y)
plt.show()
