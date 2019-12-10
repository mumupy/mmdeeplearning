#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/27 13:51
# @Author  : ganliang
# @File    : keras_fashionmnist.py
# @Desc    : 图像识别

import os

import keras
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from src.config import logger, root_path


def fashionmnist_info():
    """
    显示fashionMnist衣服分类
    :return:
    """
    fashionmnist = input_data.read_data_sets(os.path.join(root_path, "data", "fashionMNIST"), one_hot=True)

    train_images, train_labels = fashionmnist.train.images, fashionmnist.train.labels
    test_images, test_labels = fashionmnist.test.images, fashionmnist.test.labels

    logger.info("train images shape {0}".format(train_images.shape))
    logger.info("train labels shape {0}".format(train_labels.shape))
    logger.info("test images shape {0}".format(test_images.shape))
    logger.info("test labels label shape {0}".format(test_labels.shape))

    # 计算预估的最大值
    train_labels = np.argmax(train_labels, 1)
    shapesize = int(np.math.sqrt(train_images.shape[1]))
    train_images = np.reshape(train_images, newshape=(-1, shapesize, shapesize))

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    plt.figure()
    plt.imshow(train_images[0])
    plt.xlabel(class_names[int(train_labels[0])])
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.grid(False)
    plt.show()


def fashionmnist_dnn():
    """
    图片分类
    :return:
    """
    fashionmnist = input_data.read_data_sets(os.path.join(root_path, "data", "fashionMNIST"), one_hot=True)

    train_images, train_labels = fashionmnist.train.images, fashionmnist.train.labels
    test_images, test_labels = fashionmnist.test.images, fashionmnist.test.labels

    train_labels = np.argmax(train_labels, 1)
    test_labels = np.argmax(test_labels, 1)

    shapesize = int(np.math.sqrt(train_images.shape[1]))
    train_images = np.reshape(train_images, newshape=(-1, shapesize, shapesize))
    test_images = np.reshape(test_images, newshape=(-1, shapesize, shapesize))

    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=10)

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    logger.info('Test accuracy: {0}'.format(test_acc))

    predictions = model.predict(test_images)
    predictions = np.argmax(predictions, 1)
    for i in range(10):
        logger.info("predict:{0} , label:{1}".format(predictions[i], test_labels[i]))


def fashionmnist_cnn():
    fashionmnist = input_data.read_data_sets(os.path.join(root_path, "data", "fashionMNIST"), one_hot=True)

    train_images, train_labels = fashionmnist.train.images, fashionmnist.train.labels
    test_images, test_labels = fashionmnist.test.images, fashionmnist.test.labels

    shapesize = int(np.math.sqrt(train_images.shape[1]))
    train_images = np.reshape(train_images, newshape=(-1, shapesize, shapesize, 1))
    test_images = np.reshape(test_images, newshape=(-1, shapesize, shapesize, 1))

    model = keras.models.Sequential()
    model.add(keras.layers.Convolution2D(32, (3, 3), strides=1, padding="same", data_format="channels_last",
                                         input_shape=(shapesize, shapesize, 1)))
    model.add(keras.layers.Activation("relu", name="convolution1_relu"))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))

    model.add(keras.layers.Convolution2D(32, (3, 3), padding="same", data_format="channels_last", ))
    model.add(keras.layers.Activation("relu", name="convolution2_relu"))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128))
    model.add(keras.layers.Activation("relu", name="dense_128_relu"))

    model.add(keras.layers.Dense(64))
    model.add(keras.layers.Activation("relu", name="dense_64_relu"))

    model.add(keras.layers.Dense(10))
    model.add(keras.layers.Activation("softmax", name="softmax"))

    model.summary()

    model.compile(optimizer="adadelta", loss="categorical_crossentropy", metrics=["accuracy"])

    model.fit(train_images, train_labels, batch_size=128, epochs=10, verbose=2,
              validation_data=(test_images, test_labels))

    test_loss, test_acc = model.evaluate(test_images, test_labels, batch_size=32, verbose=2)
    logger.info('Test test_loss: {0}, test_acc: {1}'.format(test_loss, test_acc))

    predictions = model.predict(test_images)
    predictions = np.argmax(predictions, 1)
    for i in range(10):
        logger.info("predict:{0} , label:{1}".format(predictions[i], np.argmax(test_labels[i], 1)))


if __name__ == "__main__":
    # fashionmnist_info()
    # fashionmnist_dnn()
    fashionmnist_cnn()
