#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/12/16 14:16
# @Author  : ganliang
# @File    : keras_cifar.py
# @Desc    : cifar-10模型预测

import keras
import numpy as np

from src.config.log import logger


def keras_cifar_dnn():
    """
    cifar图片分类多层神经网络实现
    :return:
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # 图像数据预处理
    x_train = x_train / 255
    x_test = x_test / 255

    num_classes = max(np.max(y_train), np.max(y_test)) + 1
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    logger.info(x_train.shape)
    logger.info(y_train.shape)
    logger.info(x_test.shape)
    logger.info(y_test.shape)

    model = keras.models.Sequential()
    model.add(keras.layers.Convolution2D(64, (3, 3), padding="same", data_format="channels_last",
                                         activation="relu", input_shape=(32, 32, 3)))
    model.add(keras.layers.Convolution2D(64, (3, 3), padding="same", activation="relu"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Convolution2D(128, (3, 3), padding="same", activation="relu"))
    model.add(keras.layers.Convolution2D(128, (3, 3), padding="same", activation="relu"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Convolution2D(256, (3, 3), padding="same", activation="relu"))
    model.add(keras.layers.Convolution2D(256, (3, 3), padding="same", activation="relu"))
    model.add(keras.layers.Convolution2D(256, (3, 3), padding="same", activation="relu"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dense(num_classes, activation="softmax"))

    model.summary()
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=1, validation_data=(x_test, y_test))

    test_loss, test_accuracy = model.evaluate(x_test, y_test, batch_size=64, verbose=0)
    logger.info("\ntest_loss:{0},test_accuracy:{1}".format(test_loss, test_accuracy))


if __name__ == "__main__":
    keras_cifar_dnn()
