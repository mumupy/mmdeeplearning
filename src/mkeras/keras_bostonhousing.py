#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/12/16 20:55
# @Author  : ganliang
# @File    : keras_bostonhousing.py
# @Desc    : 波士顿房价预测
import os

import keras

from src.config.log import root_path, logger


def keras_bostonhousing():
    (x_train, y_train), (x_test, y_test) = keras.datasets.boston_housing.load_data(
        os.path.join(root_path, "data", "boston_housing", "boston_housing.npz"))
    logger.info(x_train.shape)
    logger.info(y_train.shape)
    logger.info(x_test.shape)
    logger.info(y_test.shape)

    # x_train = keras.utils.np_utils.normalize(x_train, 1)
    # y_train = keras.utils.np_utils.normalize(y_train, 0)[0]
    x_train = x_train / 100.
    y_train = y_train / 100.

    # 数据预处理
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(32, activation="sigmoid", input_shape=(x_train.shape[1],), use_bias=True))
    # model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    model.summary()

    model.compile(optimizer=keras.optimizers.SGD(lr=0.0001), loss="mse", metrics=[keras.metrics.categorical_accuracy()])

    result = model.fit(x_train, y_train, batch_size=16, epochs=50, verbose=1, validation_split=0.2)
    # logger.info("loss:{0}".format(loss))
    # logger.info("acc:{0}".format(acc))
    logger.info(result)


if __name__ == "__main__":
    keras_bostonhousing()
