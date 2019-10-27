#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/27 13:50
# @Author  : ganliang
# @File    : keras_mnist.py
# @Desc    : keras的手写数字识别
import os

import keras
import matplotlib.pyplot as plt

from src.config import logger, root_path


def mnist_info():
    """
    显示mnist数据
    :return:
    """
    mnist = keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data(os.path.join(root_path, "data", "mnist", "mnist.npz"))
    logger.info("train shape {0}".format(x_train.shape))
    logger.info("label shape {0}".format(y_train.shape))
    logger.info("test shape {0}".format(x_test.shape))
    logger.info("test label shape {0}".format(y_test.shape))
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_train[i])
    plt.show()


def keras_mnist():
    """
    手写数字keras识别
    :return:
    """
    mnist = keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data(os.path.join(root_path, "data", "mnist", "mnist.npz"))
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        # keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)

    model.evaluate(x_test, y_test, verbose=2)


if __name__ == "__main__":
    mnist_info()
    keras_mnist()
