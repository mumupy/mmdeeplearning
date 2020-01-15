#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/20 18:53
# @Author  : ganliang
# @File    : ndchar.py
# @Desc    : 数据字符
import numpy as np

print(np.char.add("baby", "mm"))
print(np.char.add(['hello', 'hi'], [' abc', ' xyz']))

print(np.char.multiply("baby", 10))
print(np.char.center("baby", 10, "."))

print(np.char.capitalize("baby"))
print(np.char.title("baby lover cws"))
print(np.char.lower("baby lover cws"))
print(np.char.upper("baby lover cws"))

print(np.char.split("baby lover cws"))
print(np.char.splitlines("baby lover cws"))

print(np.char.strip(" baby lover cws "))
print(np.char.join(":", " baby lover cws "))

print(np.char.replace("baby lover cws", "cws", ""))
print(np.char.encode("baby lover cws", "utf-8"))
print(np.char.decode("baby lover cws", "cp500"))
