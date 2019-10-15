#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/22 10:42
# @Author  : ganliang
# @File    : __init__.py.py
# @Desc    : tensorflow深度学习框架

import os

from src.config import logger


def tensorboard():
    """
    启动tensorvoard
    :return:
    """
    stdin, stdout = os.popen("tensorboard --logdir tensorboard")
    for line in stdout.readlines():
        logger.info(line)


if __name__ == "__main__":
    tensorboard()
