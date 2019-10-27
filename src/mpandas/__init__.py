#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/22 10:41
# @Author  : ganliang
# @File    : __init__.py.py
# @Desc    : pandas测试
import numpy as np
import pandas as pd

from src.config import logger


def pd_serise():
    series = pd.Series([1, 3, 5, np.nan, 6, 8])
    logger.info("Series:\n{0}".format(series))


def pd_date_range():
    date_ranges = pd.date_range("2019-10-01", periods=6, freq="D")
    logger.info("date_ranges:\n{0}".format(date_ranges))


def pd_dataframe():
    pd_frame = pd.DataFrame(np.random.rand(6, 4), index=["R1", "R2", "R3", "R4", "R5", "R6"],
                            columns=["A", "B", "C", "D"])
    logger.info("pd_frame:\n{0}".format(pd_frame))
    logger.info("pd_frame.head(1):\n{0}".format(pd_frame.head(1)))
    logger.info("pd_frame.tail(1):\n{0}".format(pd_frame.tail(1)))
    logger.info("pd_frame.dtypes:\n{0}".format(pd_frame.dtypes))
    logger.info("pd_frame.shape:\n{0}".format(pd_frame.shape))
    logger.info("pd_frame.to_numpy():\n{0}".format(pd_frame.to_numpy()))
    logger.info("pd_frame.describe():\n{0}".format(pd_frame.describe()))
    logger.info("pd_frame.T:\n{0}".format(pd_frame.T))
    logger.info(
        "pd_frame.sort_index(axis=1, ascending=False):\n{0}".format(pd_frame.sort_index(axis=1, ascending=False)))
    logger.info("pd_frame.sort_values(by='B'):\n{0}".format(pd_frame.sort_values(by='B')))
    logger.info("pd_frame['A']:\n{0}".format(pd_frame["A"]))
    logger.info("pd_frame[0:1]:\n{0}".format(pd_frame[0:1]))


if __name__ == "__main__":
    # pd_serise()
    # pd_date_range()
    pd_dataframe()
