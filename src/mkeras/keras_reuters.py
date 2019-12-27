#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/12/23 10:17
# @Author  : ganliang
# @File    : keras_reuters.py
# @Desc    : 路透社新闻分类
import os

import keras
from keras.datasets import reuters
from keras.preprocessing import sequence, text

from mkeras import keras_history_plotcurve
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
    num_vocab = max(max([max(x) for x in X_train]), max([max(x) for x in X_test])) + 1

    logger.info("num_words {0}".format(num_words))
    logger.info("num_classify {0}".format(num_classify))
    logger.info("num_voc {0}".format(num_vocab))


def keras_reuters_mlp(num_words=None, maxlen=None, num_categorical=None, batch_size=32, epochs=10,
                      mode=None):
    """
    使用mlp多层感知机模式进行情感评估,同时对比自归一化mlp和常规mlp的性能对比
    :return:
    """
    (X_train, y_train), (X_test, y_test) = reuters.load_data(
        path=os.path.join(root_path, "data", "reuters", "reuters.npz"), num_words=num_words)

    if not num_words: num_words = max(max([max(x) for x in X_train]), max([max(x) for x in X_test])) + 1
    if not maxlen: maxlen = max(max([len(x) for x in X_train]), max([len(x) for x in X_test])) + 1
    if not num_categorical: num_categorical = max(max(y_train), max(y_test)) + 1

    tokenizer = text.Tokenizer(num_words=num_words)
    X_train = tokenizer.sequences_to_matrix(X_train)
    y_train = keras.utils.to_categorical(y_train, num_categorical)
    X_test = tokenizer.sequences_to_matrix(X_test)
    y_test = keras.utils.to_categorical(y_test, num_categorical)

    input = keras.layers.Input(shape=(num_words,))
    # 自归一化snn
    if mode == "self-normalizing":
        x = keras.layers.Dense(512, activation=keras.activations.selu, kernel_initializer="lecun_normal")(input)
        x = keras.layers.AlphaDropout(0.1)(x)

        x = keras.layers.Dense(256, activation="selu", kernel_initializer="lecun_normal")(x)
        x = keras.layers.AlphaDropout(0.1)(x)

        x = keras.layers.Dense(128, activation="selu", kernel_initializer="lecun_normal")(x)
        x = keras.layers.AlphaDropout(0.1)(x)
    else:
        x = keras.layers.Dense(512, activation="relu", kernel_initializer="glorot_normal")(input)
        x = keras.layers.BatchNormalization()(x)
        # x = keras.layers.Dropout(0.4)(x)

        x = keras.layers.Dense(256, activation="relu", kernel_initializer="glorot_normal")(x)
        x = keras.layers.BatchNormalization()(x)
        # x = keras.layers.Dropout(0.4)(x)

        x = keras.layers.Dense(128, activation="relu", kernel_initializer="glorot_normal")(x)
        x = keras.layers.BatchNormalization()(x)
        # x = keras.layers.Dropout(0.4)(x)

    x = keras.layers.Dense(num_categorical, activation="softmax")(x)

    model = keras.models.Model(inputs=input, outputs=x)
    model.summary()

    model.compile(optimizer="adadelta", loss="categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)
    keras_history_plotcurve(history)

    score = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
    logger.info('Test loss:{0}'.format(score[0]))
    logger.info('Test accuracy:{0}'.format(score[1]))


def keras_reuters_cnn():
    (X_train, y_train), (X_test, y_test) = reuters.load_data(
        path=os.path.join(root_path, "data", "reuters", "reuters.npz"))

    num_words = max(max([len(x) for x in X_train]), max([len(x) for x in X_test])) + 1
    num_classify = max(max(y_train), max(y_test)) + 1
    num_vocab = max(max([max(x) for x in X_train]), max([max(x) for x in X_test])) + 1

    X_train = sequence.pad_sequences(X_train, maxlen=num_words)
    y_train = keras.utils.to_categorical(y_train, num_classify)
    X_test = sequence.pad_sequences(X_test, maxlen=num_words)
    y_test = keras.utils.to_categorical(y_test, num_classify)

    input = keras.layers.Input(shape=(num_words,))
    x = keras.layers.Embedding(input_dim=num_vocab + 1, output_dim=128)(input)
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
    history = model.fit(X_train, y_train, batch_size=64, epochs=10, verbose=1, validation_data=(X_test, y_test))
    keras_history_plotcurve(history)


def keras_reuters_rnn(num_words=None, maxlen=None, num_categorical=None):
    """
    使用rnn进行新闻分类
    :return:
    """
    (X_train, y_train), (X_test, y_test) = reuters.load_data(
        path=os.path.join(root_path, "data", "reuters", "reuters.npz"), num_words=num_words)

    if not num_words: num_words = max(max([max(x) for x in X_train]), max([max(x) for x in X_test])) + 1
    if not maxlen: maxlen = max(max([len(x) for x in X_train]), max([len(x) for x in X_test])) + 1
    if not num_categorical: num_categorical = max(max(y_train), max(y_test)) + 1

    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    y_train = keras.utils.to_categorical(y_train, num_categorical)
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
    y_test = keras.utils.to_categorical(y_test, num_categorical)

    input = keras.layers.Input(shape=(maxlen,))
    x = keras.layers.Embedding(input_dim=num_words, output_dim=128)(input)
    x = keras.layers.SimpleRNN(32)(x)

    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dense(num_categorical, activation="softmax")(x)
    model = keras.models.Model(inputs=input, outputs=x)
    model.summary()

    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(X_train, y_train, batch_size=64, epochs=10, verbose=1, validation_data=(X_test, y_test))
    keras_history_plotcurve(history)


def keras_reuters_lstm(num_words=None, maxlen=None, num_categorical=None):
    """
    使用lstm进行新闻分类,lstm比simpleRNN擅长处理长序列。lstm包含输入门、输出门和重置门
    :return:
    """
    (X_train, y_train), (X_test, y_test) = reuters.load_data(
        path=os.path.join(root_path, "data", "reuters", "reuters.npz"), num_words=num_words)

    if not num_words: num_words = max(max([max(x) for x in X_train]), max([max(x) for x in X_test])) + 1
    if not maxlen: maxlen = max(max([len(x) for x in X_train]), max([len(x) for x in X_test])) + 1
    if not num_categorical: num_categorical = max(max(y_train), max(y_test)) + 1

    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    y_train = keras.utils.to_categorical(y_train, num_categorical)
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
    y_test = keras.utils.to_categorical(y_test, num_categorical)

    input = keras.layers.Input(shape=(maxlen,))
    x = keras.layers.Embedding(input_dim=num_words, output_dim=128)(input)
    x = keras.layers.LSTM(32)(x)

    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dense(num_categorical, activation="softmax")(x)
    model = keras.models.Model(inputs=input, outputs=x)
    model.summary()

    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(X_train, y_train, batch_size=64, epochs=10, verbose=1, validation_data=(X_test, y_test))
    keras_history_plotcurve(history)


def keras_reuters_gru(num_words=None, maxlen=None, num_categorical=None):
    """
    使用gru门控循环单元进行新闻分类，gru包含输入门、输出门、重置门和更新门。
    :return:
    """
    (X_train, y_train), (X_test, y_test) = reuters.load_data(
        path=os.path.join(root_path, "data", "reuters", "reuters.npz"), num_words=num_words)

    if not num_words: num_words = max(max([max(x) for x in X_train]), max([max(x) for x in X_test])) + 1
    if not maxlen: maxlen = max(max([len(x) for x in X_train]), max([len(x) for x in X_test])) + 1
    if not num_categorical: num_categorical = max(max(y_train), max(y_test)) + 1

    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    y_train = keras.utils.to_categorical(y_train, num_categorical)
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
    y_test = keras.utils.to_categorical(y_test, num_categorical)

    input = keras.layers.Input(shape=(maxlen,))
    x = keras.layers.Embedding(input_dim=num_words, output_dim=128)(input)
    x = keras.layers.GRU(32)(x)

    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dense(num_categorical, activation="softmax")(x)
    model = keras.models.Model(inputs=input, outputs=x)
    model.summary()

    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(X_train, y_train, batch_size=64, epochs=10, verbose=1, validation_data=(X_test, y_test))
    keras_history_plotcurve(history)


if __name__ == "__main__":
    # keras_reuters_info()
    keras_reuters_mlp(num_words=10000, maxlen=500)
    # keras_reuters_cnn()
    # keras_reuters_rnn(num_words=10000,maxlen=500)
    # keras_reuters_lstm()
    # keras_reuters_gru()
