#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/30 8:36
# @Author  : ganliang
# @File    : onehot.py
# @Desc    : onehot编码

from sklearn import preprocessing
from src.config import logger


def onehot():
    """
    One-Hot编码，又称为一位有效编码，主要是采用N位状态寄存器来对N个状态进行编码，每个状态都由他独立的寄存器位，并且在任意时候只有一位有效。
    One-Hot编码是分类变量作为二进制向量的表示。这首先要求将分类值映射到整数值。然后，每个整数值被表示为二进制向量，除了整数的索引之外，它都
    是零值，它被标记为1。
    :return:
    """
    enc = preprocessing.OneHotEncoder()
    enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])  # 这里一共有4个数据，3种特征
    array = enc.transform([[0, 1, 2]]).toarray()  # 这里使用一个新的数据来测试
    print(array)


if __name__ == "__main__":
    onehot()
