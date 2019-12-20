#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/12/19 20:41
# @Author  : ganliang
# @File    : keras_bottleneck.py
# @Desc    : bottleneck瓶颈网络
import os

import keras
import numpy as np
from keras.datasets import mnist

from src.config.log import root_path


def keras_bottleneck_model():
    """
    训练卷积网络的参数，然后将卷积网络的参数直接使用预测过滤器数据，然后将过滤器数据用于新的模型测试
    :return:
    """
    (X_train, y_train), (X_test, y_test) = mnist.load_data(os.path.join(root_path, "data", "mnist", "mnist.npz"))
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255
    X_train = X_train.reshape(X_train.shape + (1,))
    X_test = X_test.reshape(X_test.shape + (1,))

    Y_train = keras.utils.np_utils.to_categorical(y_train, 10)
    Y_test = keras.utils.np_utils.to_categorical(y_test, 10)

    model = keras.models.Sequential()
    model.add(keras.layers.Convolution2D(32, (3, 3), padding='valid', input_shape=(28, 28, 1),
                                         data_format="channels_last", activation="relu"))
    model.add(keras.layers.Convolution2D(62, (3, 3), padding="valid", activation="relu",
                                         data_format="channels_last"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), name="bottleneck_layer"))

    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Dense(10, activation="softmax"))
    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    os.makedirs(os.path.join(root_path, "tmp", "mnist", "bottleneck"), exist_ok=True)

    model.fit(X_train, Y_train, batch_size=64, epochs=10, verbose=1, validation_data=(X_test, Y_test))

    # 保存卷积权重
    bottleneck_model = keras.models.Sequential()

    for layer in model.layers:
        bottleneck_model.add(layer)
        if layer.name == "bottleneck_layer": break

    bottleneck_model.save(os.path.join(root_path, "tmp", "mnist", "bottleneck", "mnist_bottleneck_model.h5"))


def keras_bottleneck_predict():
    """
    获取bottleneck预测的过滤器数据对数据进行评估，将评估的数据保存到文件中
    :return:
    """
    # 使用自定义的mnist模型
    model = keras.models.load_model(os.path.join(root_path, "tmp", "mnist", "bottleneck", "mnist_bottleneck_model.h5"))
    model.summary()

    # 加载数据
    (X_train, y_train), (X_test, y_test) = mnist.load_data(os.path.join(root_path, "data", "mnist", "mnist.npz"))
    X_train = np.reshape(X_train, X_train.shape + (1,)) / 255.
    X_test = np.reshape(X_test, X_test.shape + (1,)) / 255.

    # 数据增强,保存训练数据集和验证数据集
    # train_datagen = image.ImageDataGenerator(rotation_range=0.2, width_shift_range=0.2, height_shift_range=0.2,
    #                                          rescale=1 / 255., horizontal_flip=True)
    # train_datagen.fit(X_train)
    # train_generator = train_datagen.flow(X_train, batch_size=128, shuffle=True)
    # bottleneck_features_train = model.predict_generator(train_generator, X_train.shape[0] // 128, verbose=1)
    os.makedirs(os.path.join(root_path, "tmp", "mnist", "bottleneck"), exist_ok=True)

    bottleneck_features_train = model.predict(X_train, batch_size=128, verbose=1)
    np.save(os.path.join(root_path, "tmp", "mnist", "bottleneck", "bottleneck_features_train.npy"),
            bottleneck_features_train, allow_pickle=True)

    # validation_datagen = image.ImageDataGenerator(rotation_range=0.2, width_shift_range=0.2, height_shift_range=0.2)
    # validation_datagen.fit(X_test)
    # validation_generator = validation_datagen.flow(X_test, batch_size=128, shuffle=True)
    # bottleneck_features_validation = model.predict_generator(validation_generator, X_test.shape[0] // 128, verbose=1)
    bottleneck_features_validation = model.predict(X_test, batch_size=128, verbose=1)
    np.save(os.path.join(root_path, "tmp", "mnist", "bottleneck", "bottleneck_features_validation.npy"),
            bottleneck_features_validation, allow_pickle=True)


def keras_bottleneck_train():
    """
    加载保存的数据
    :return:
    """
    (X_train, y_train), (X_test, y_test) = mnist.load_data(os.path.join(root_path, "data", "mnist", "mnist.npz"))

    train_data = np.load(os.path.join(root_path, "tmp", "mnist", "bottleneck", "bottleneck_features_train.npy"))
    train_labels = keras.utils.np_utils.to_categorical(y_train, 10)[:train_data.shape[0]]

    validation_data = np.load(
        os.path.join(root_path, "tmp", "mnist", "bottleneck", "bottleneck_features_validation.npy"))
    validation_labels = keras.utils.np_utils.to_categorical(y_test, 10)[:validation_data.shape[0]]

    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=train_data.shape[1:]))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Dense(10, activation='softmax'))
    model.summary()

    model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels, epochs=10, batch_size=128,
              validation_data=(validation_data, validation_labels))
    model.save_weights(os.path.join(root_path, "tmp", "mnist", "bottleneck", "bottleneck_fc_model.h5"))


if __name__ == "__main__":
    # keras_bottleneck_model()
    # keras_bottleneck_predict()
    keras_bottleneck_train()
