#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/26 14:41
# @Author  : ganliang
# @File    : modelevaluation.py
# @Desc    : 模型选择和验证 https://scikit-learn.org/stable/model_selection.html#model-selection
import numpy as np
from sklearn.model_selection import LeaveOneOut

from src.config import logger


def leave_oneout():
    """
    留一发 就是循环数据集次数N 每次留取一个数据作为测试数据集，(N-1)数据为训练数据集。
    :return:
    """
    X = np.array([[1, 2, 3, 4],
                  [11, 12, 13, 14],
                  [21, 22, 23, 24],
                  [31, 32, 33, 34]])
    y = np.array([1, 1, 0, 0])
    loo = LeaveOneOut()
    for train_index, test_index in loo.split(X, y):
        logger.info("Train Index:\n %s" % train_index)
        logger.info("Test Index:\n %s" % test_index)
        logger.info("X_train:\n %s" % X[train_index])
        logger.info("X_test:\n %s" % X[test_index])
        logger.info("")


if __name__ == "__main__":
    leave_oneout()
