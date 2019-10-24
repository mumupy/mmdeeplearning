#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/24 19:19
# @Author  : ganliang
# @File    : __init__.py.py
# @Desc    : seaborn绘图软件实例

import seaborn as sns
import matplotlib.pyplot as plt

sns.set()


fmri = sns.load_dataset("fmri")
# ax = sns.lineplot(x="timepoint", y="signal", data=fmri)
ax = sns.lineplot(x="timepoint", y="signal", hue="event",data=fmri)
plt.show()
