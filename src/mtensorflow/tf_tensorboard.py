#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/22 16:33
# @Author  : ganliang
# @File    : tf_tensorboard.py
# @Desc    : tensorboard图形化展示

from src.config import logger

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def tensorboard():
    """
    启动tensorvoard
    :return:
    """
    stdin, stdout = os.popen("tensorboard --logdir ./tensorboard")
    for line in stdout.readlines():
        logger.info(line)


if __name__ == "__main__":
    tensorboard()
