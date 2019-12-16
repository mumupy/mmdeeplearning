#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/27 14:58
# @Author  : ganliang
# @File    : keras_imdb.py
# @Desc    : imdb电影评价
import os

import keras
from keras.preprocessing import sequence

imdb = keras.datasets.imdb
from src.config import logger, root_path


def imdb_info():
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(
        os.path.join(root_path, "data", "imdb", "imdb.npz"))
    logger.info("train data shape {0}".format(train_data.shape))
    logger.info("train labels shape {0}".format(train_labels.shape))
    logger.info("test data shape {0}".format(test_data.shape))
    logger.info("test label shape {0}".format(test_labels.shape))

    # 一个映射单词到整数索引的词典
    word_index = imdb.get_word_index(os.path.join(root_path, "data", "imdb", "imdb_word_index.json"))

    # 保留第一个索引
    word_index = {k: (v + 3) for k, v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2  # unknown
    word_index["<UNUSED>"] = 3

    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    def decode_review(text):
        return ' '.join([reverse_word_index.get(i, '?') for i in text])

    logger.info(decode_review(train_data[0]))


def imdb_keras():
    """
    影评
    :return:
    """
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(
        os.path.join(root_path, "data", "imdb", "imdb.npz"))
    vocab_size = 100000

    word_index = imdb.get_word_index(os.path.join(root_path, "data", "imdb", "imdb_word_index.json"))
    word_index = {k: (v + 3) for k, v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2  # unknown
    word_index["<UNUSED>"] = 3

    train_data = sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding='pre', maxlen=256)
    test_data = sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding='post', maxlen=256)

    model = keras.models.Sequential()
    model.add(keras.layers.Embedding(vocab_size, 16))

    model.add(keras.layers.Conv1D(32, 5, activation="relu"))
    model.add(keras.layers.MaxPool1D())
    model.add(keras.layers.Dropout(0.4))

    model.add(keras.layers.Conv1D(32, 5, activation="relu"))
    model.add(keras.layers.GlobalMaxPool1D())
    model.add(keras.layers.Dropout(0.4))

    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.summary()
    model.compile(optimizer="rmsprop", loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels, epochs=10, batch_size=64, validation_data=(test_data, test_labels),
              verbose=1)

    # 评估模型
    results = model.evaluate(test_data, test_labels, verbose=1)

    logger.info(results)


if __name__ == "__main__":
    # imdb_info()
    imdb_keras()
