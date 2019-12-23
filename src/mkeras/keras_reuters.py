#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/12/23 10:17
# @Author  : ganliang
# @File    : keras_reuters.py
# @Desc    : 路透社新闻分类
import os

import keras
from keras.datasets import reuters
from keras.preprocessing import sequence

from src.config.log import logger, root_path


def keras_reuters_info():
    (X_train, y_train), (X_test, y_test) = reuters.load_data(
        path=os.path.join(root_path, "data", "reuters", "reuters.npz"), skip_top=0, maxlen=None,
        test_split=0.2, seed=113, start_char=1, oov_char=2,
        index_from=3)
    logger.info(X_train.shape)
    logger.info(y_train.shape)
    logger.info(X_test.shape)
    logger.info(y_test.shape)

    word_index = reuters.get_word_index(os.path.join(root_path, "data", "reuters", "reuters_word_index.json"))
    logger.info(word_index)

    num_words = max(max([len(x) for x in X_train]), max([len(x) for x in X_test])) + 1
    num_classify = max(max(y_train), max(y_test)) + 1
    num_voc = max(max([max(x) for x in X_train]), max([max(x) for x in X_test])) + 1

    logger.info("num_words {0}".format(num_words))
    logger.info("num_classify {0}".format(num_classify))
    logger.info("num_voc {0}".format(num_voc))


def keras_reuters_mlp():
    """
    使用mlp多层感知机模式进行情感评估
    :return:
    """
    (X_train, y_train), (X_test, y_test) = reuters.load_data(
        path=os.path.join(root_path, "data", "reuters", "reuters.npz"))


def keras_reuters_cnn():
    (X_train, y_train), (X_test, y_test) = reuters.load_data(
        path=os.path.join(root_path, "data", "reuters", "reuters.npz"))

    num_words = max(max([len(x) for x in X_train]), max([len(x) for x in X_test])) + 1
    num_classify = max(max(y_train), max(y_test)) + 1
    num_voc = max(max([max(x) for x in X_train]), max([max(x) for x in X_test])) + 1

    X_train = sequence.pad_sequences(X_train, maxlen=num_words)
    y_train = keras.utils.to_categorical(y_train, num_classify)
    X_test = sequence.pad_sequences(X_test, maxlen=num_words)
    y_test = keras.utils.to_categorical(y_test, num_classify)

    input = keras.layers.Input(shape=(num_words,))
    x = keras.layers.Embedding(input_dim=num_voc + 1, output_dim=128)(input)
    x = keras.layers.Convolution1D(32, 5, activation="relu")(x)
    x = keras.layers.Convolution1D(32, 5, activation="relu")(x)
    x = keras.layers.MaxPooling1D(5)(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Convolution1D(32, 5, activation="relu")(x)
    x = keras.layers.Convolution1D(32, 5, activation="relu")(x)
    x = keras.layers.MaxPooling1D(5)(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dense(num_classify, activation="softmax")(x)

    model = keras.models.Model(inputs=input, outputs=x)
    model.summary()

    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, batch_size=64, epochs=10, verbose=1, validation_data=(X_test, y_test))


def keras_reuters_rnn():
    """
    使用rnn进行新闻分类
    :return:
    """
    (X_train, y_train), (X_test, y_test) = reuters.load_data(
        path=os.path.join(root_path, "data", "reuters", "reuters.npz"))

    num_words = max(max([len(x) for x in X_train]), max([len(x) for x in X_test])) + 1
    num_classify = max(max(y_train), max(y_test)) + 1
    num_voc = max(max([max(x) for x in X_train]), max([max(x) for x in X_test])) + 1

    X_train = sequence.pad_sequences(X_train, maxlen=num_words)
    y_train = keras.utils.to_categorical(y_train, num_classify)
    X_test = sequence.pad_sequences(X_test, maxlen=num_words)
    y_test = keras.utils.to_categorical(y_test, num_classify)

    input = keras.layers.Input(shape=(num_words,))
    x = keras.layers.Embedding(input_dim=num_voc, output_dim=128)(input)
    x = keras.layers.SimpleRNN(32)(x)

    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dense(num_classify, activation="softmax")(x)
    model = keras.models.Model(inputs=input, outputs=x)
    model.summary()

    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, batch_size=64, epochs=10, verbose=1, validation_data=(X_test, y_test))


if __name__ == "__main__":
    # keras_reuters_info()
    # keras_reuters_cnn()
    keras_reuters_rnn()
