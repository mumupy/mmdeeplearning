#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/27 13:50
# @Author  : ganliang
# @File    : keras_mnist.py
# @Desc    : keras的手写数字识别
import os

import keras
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import Convolution2D, Activation, MaxPooling2D, Dropout, Flatten, Dense, K, np
from keras.utils import np_utils

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


def mnist_dnn():
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
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)

    model.evaluate(x_test, y_test, verbose=2)


def mnist_conv():
    """
    使用卷积神经网络训练图片分类
    :return:
    """
    batch_size = 128
    nb_classes = 10  # 分类数
    nb_epoch = 12  # 训练轮数
    # 输入图片的维度
    img_rows, img_cols = 28, 28
    # 卷积滤镜的个数
    nb_filters = 32
    # 最大池化，池化核大小
    pool_size = (2, 2)
    # 卷积核大小
    kernel_size = (3, 3)

    (X_train, y_train), (X_test, y_test) = mnist.load_data(os.path.join(root_path, "data", "mnist", "mnist.npz"))

    if K.image_dim_ordering() == 'th':
        # 使用 Theano 的顺序：(conv_dim1, channels, conv_dim2, conv_dim3)
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        # 使用 TensorFlow 的顺序：(conv_dim1, conv_dim2, conv_dim3, channels)
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    model = keras.models.Sequential()
    model.add(
        Convolution2D(nb_filters, kernel_size, padding='valid', input_shape=input_shape, data_format="channels_last"))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, kernel_size))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
              verbose=1, validation_data=(X_test, Y_test))

    test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=0)

    logger.info('Test accuracy: {0} {1}'.format(test_loss, test_acc))

    predictions = model.predict(X_test)
    predictions = np.argmax(predictions, 1)
    labels = np.argmax(Y_test, 1)
    for i in range(10):
        logger.info("predict:{0} , label:{1}".format(predictions[i], labels[i]))


if __name__ == "__main__":
    # mnist_info()
    # mnist_dnn()
    mnist_conv()
