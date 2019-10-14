#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/22 22:46
# @Author  : ganliang
# @File    : logging.py
# @Desc    : 日志配置

import logging.config
import os
import traceback

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
config_path = os.path.dirname(os.path.abspath(__file__))
config_file = os.path.join(root_path, 'logging.conf')


def get_log_path():
    """
    获取日志路径
    :return:
    """
    try:
        import configparser
        configparser = configparser.ConfigParser()
        configparser.read(config_file)

        file_handler_args = configparser.get("handler_fileHandler", "args")
        logargs = str(file_handler_args).replace("(", "").replace(")", "").split(",")
        logfile = os.path.join(root_path, str(logargs[0]).replace("'", ""))

        # 创建日志目录
        logdir = os.path.dirname(logfile)
        if not os.path.exists(logdir):
            os.makedirs(logdir)

        logargs.pop(0)
        logargs.insert(0, "'{0}'".format(logfile))
        logargs = "({0})".format(",".join(logargs))
        configparser.set("handler_fileHandler", "args", logargs)
    except Exception as ex:
        traceback.print_stack()
        return config_file
    else:
        return configparser


logging.config.fileConfig(get_log_path())

logger = logging.getLogger(__file__)

if __name__ == '__main__':
    logger.debug('Hello Debug Webben')
    logger.info('Hello Info Webben')
    logger.error('Hello Error Webben')

__all__ = ["logger", "root_path"]
