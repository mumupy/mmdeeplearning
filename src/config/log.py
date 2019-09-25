#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/22 22:46
# @Author  : ganliang
# @File    : logging.py
# @Desc    : 日志配置
import logging.config
import os

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
config_path = os.path.dirname(os.path.abspath(__file__))

logging.config.fileConfig(os.path.join(root_path, 'logging.conf'))

logger = logging.getLogger(__file__)

if __name__ == '__main__':
    logger.debug('Hello Debug Webben')
    logger.info('Hello Info Webben')
    logger.error('Hello Error Webben')

__all__ = ["logger","root_path"]
