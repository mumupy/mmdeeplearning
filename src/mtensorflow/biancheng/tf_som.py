#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/26 12:42
# @Author  : ganliang
# @File    : tf_som.py
# @Desc    : 自组织映射（SOM网络，也被称为 Kohonen 网络或者胜者独占单元（WTU），在大脑中，不同的感官输入以拓扑顺序的方式呈现，是受人脑特征启发而提出的一种非常特殊的神经网络,
# SOM 是一个计算密集型网络，因此对于大型数据集并不实用，不过，该算法很容易理解，很容易发现输入数据之间的相似性。因此被广泛用于图像分割和自然语言处理的单词相似性映射中。


import tensorflow as tf
import numpy as np
import  matplotlib.pyplot as plt