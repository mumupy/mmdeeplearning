#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/12/19 21:02
# @Author  : ganliang
# @File    : keras_finetune.py
# @Desc    : 模型预训练
import os

import keras
from keras.datasets import mnist

from src.config.log import root_path


def keras_finetune_model():
    """
    :return:
    """
    (X_train, y_train), (X_test, y_test) = mnist.load_data(os.path.join(root_path, "data", "mnist", "mnist.npz"))
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255
    X_train = X_train.reshape(X_train.shape + (1,))
    X_test = X_test.reshape(X_test.shape + (1,))

    Y_train = keras.utils.np_utils.to_categorical(y_train, 10)
    Y_test = keras.utils.np_utils.to_categorical(y_test, 10)

    input = keras.layers.Input(shape=(28, 28, 1), name="input")
    x = keras.layers.Convolution2D(32, (3, 3), padding='same', data_format="channels_last", activation="relu")(input)
    x = keras.layers.Convolution2D(32, (3, 3), padding='same', data_format="channels_last", activation="relu")(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2), name="maxpooling2d_bottleneck_layer1")(x)

    x = keras.layers.Convolution2D(64, (3, 3), padding='same', data_format="channels_last", activation="relu")(x)
    x = keras.layers.Convolution2D(64, (3, 3), padding='same', data_format="channels_last", activation="relu")(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2), name="maxpooling2d_bottleneck_layer2")(x)

    x = keras.layers.Dropout(0.25)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.25)(x)
    x = keras.layers.Dense(10, activation="softmax")(x)
    model = keras.models.Model(inputs=input, outputs=x)
    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    os.makedirs(os.path.join(root_path, "tmp", "mnist", "finetune"), exist_ok=True)

    model.fit(X_train, Y_train, batch_size=64, epochs=10, verbose=1, validation_data=(X_test, Y_test))

    # 保存卷积权重
    finetune_cnn_data = None
    for layer in model.layers:
        if finetune_cnn_data is None:
            finetune_cnn_data = layer.input
        else:
            finetune_cnn_data = layer(finetune_cnn_data)
        if layer.name == "maxpooling2d_bottleneck_layer2": break
    finetune_cnn_model = keras.models.Model(inputs=input, outputs=finetune_cnn_data)
    finetune_cnn_model.summary()
    finetune_cnn_model.save(os.path.join(root_path, "tmp", "mnist", "finetune", "mnist_finetune_model.h5"))


def keras_finetune_train():
    """
    从预训练的模型进行编译预测
    :return:
    """
    (X_train, y_train), (X_test, y_test) = mnist.load_data(os.path.join(root_path, "data", "mnist", "mnist.npz"))
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255
    X_train = X_train.reshape(X_train.shape + (1,))
    X_test = X_test.reshape(X_test.shape + (1,))

    Y_train = keras.utils.np_utils.to_categorical(y_train, 10)
    Y_test = keras.utils.np_utils.to_categorical(y_test, 10)

    # 加载预训练的模型
    model = keras.models.load_model(os.path.join(root_path, "tmp", "mnist", "finetune", "mnist_finetune_model.h5"))
    for layer in model.layers: layer.trainable = False
    model.summary()

    # 添加自己的模型
    top_model = keras.models.Sequential()
    for layer in model.layers: top_model.add(layer)
    top_model.add(keras.layers.Flatten(input_shape=X_train.shape[1:]))
    top_model.add(keras.layers.Dense(128, activation='relu'))
    top_model.add(keras.layers.Dropout(0.25))
    top_model.add(keras.layers.Dense(10, activation='softmax'))
    top_model.summary()

    top_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    top_model.fit(X_train, Y_train, epochs=10, batch_size=128, validation_data=(X_test, Y_test))
    top_model.save_weights(os.path.join(root_path, "tmp", "mnist", "finetune", "finetune_fc_model.h5"))


if __name__ == "__main__":
    # keras_finetune_model()
    keras_finetune_train()
