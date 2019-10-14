#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/26 14:41
# @Author  : ganliang
# @File    : modelevaluation.py
# @Desc    : 模型选择和验证 https://scikit-learn.org/stable/model_selection.html#model-selection
import numpy as np
from sklearn.model_selection import LeaveOneOut, LeavePOut, KFold, GroupKFold

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
    y = np.array([1, 5, 0, 0])
    leave_oneout = LeaveOneOut()
    logger.info(leave_oneout.get_n_splits(X))
    for train_index, test_index in leave_oneout.split(X, y):
        logger.info("Train Index:\n %s" % train_index)
        logger.info("Test Index:\n %s" % test_index)
        logger.info("X_train:\n %s" % X[train_index])
        logger.info("X_test:\n %s" % X[test_index])
        logger.info("y_train:\n %s" % y[train_index])
        logger.info("y_test:\n %s" % y[test_index])
        logger.info("\n\n")


def leave_pout():
    """
    留p发 就是循环数据集次数N 每次留取p个数据作为测试数据集，(N-p)数据为训练数据集。
    :return:
    """
    X = np.array([[1, 2, 3, 4],
                  [11, 12, 13, 14],
                  [21, 22, 23, 24],
                  [31, 32, 33, 34]])
    y = np.array([1, 5, 0, 0])
    # 当p=1的时候和LeaveOneOut一样
    leave_pout = LeavePOut(p=2)
    logger.info(leave_pout.get_n_splits(X))
    for train_index, test_index in leave_pout.split(X, y):
        logger.info("Train Index:\n %s" % train_index)
        logger.info("Test Index:\n %s" % test_index)
        logger.info("X_train:\n %s" % X[train_index])
        logger.info("X_test:\n %s" % X[test_index])
        logger.info("y_train:\n %s" % y[train_index])
        logger.info("y_test:\n %s" % y[test_index])
        logger.info("\n\n")


def k_fold():
    """
    k折交叉验证法，将数据平均分成k份，留取一份作为测试数据集，其他的为训练数据集。当k等于数据集的数量的时候等同于留一发
    :return:
    """

    X = np.array([[1, 2, 3, 4],
                  [11, 12, 13, 14],
                  [21, 22, 23, 24],
                  [31, 32, 33, 34]])
    y = np.array([1, 5, 0, 0])

    k_fold = KFold(n_splits=2, shuffle=False, random_state=None)

    for train_index, test_index in k_fold.split(X, y):
        logger.info("Train Index:\n %s" % train_index)
        logger.info("Test Index:\n %s" % test_index)
        logger.info("X_train:\n %s" % X[train_index])
        logger.info("X_test:\n %s" % X[test_index])
        logger.info("\n\n")


def group_k_fold():
    """
    :return:
    """

    X = np.array([[1, 2, 3, 4],
                  [11, 12, 13, 14],
                  [21, 22, 23, 24],
                  [31, 32, 33, 34]])
    y = np.array([1, 5, 0, 0])
    groups = np.array([0, 0, 1, 1])

    group_k_fold = GroupKFold(n_splits=2)

    for train_index, test_index in group_k_fold.split(X, y,groups):
        logger.info("Train Index:\n %s" % train_index)
        logger.info("Test Index:\n %s" % test_index)
        logger.info("X_train:\n %s" % X[train_index])
        logger.info("X_test:\n %s" % X[test_index])
        logger.info("\n\n")


if __name__ == "__main__":
    # leave_oneout()
    # leave_pout()
    k_fold()
