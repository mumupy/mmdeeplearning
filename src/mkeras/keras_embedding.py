#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/12/20 16:59
# @Author  : ganliang
# @File    : keras_embedding.py
# @Desc    : 词嵌入 使用embedding+conv1d做新闻分类
import os

import keras
import numpy as np
from keras.preprocessing import text, sequence


def keras_news20_texts(news_path):
    """
    获取新闻的文本和新闻类型
    :return:
    """
    texts, labels, labels_index = [], [], {}

    for newsgroup in os.listdir(news_path):
        label_id = len(labels_index)
        labels_index.setdefault(newsgroup, label_id)
        for news_file in os.listdir(os.path.join(news_path, newsgroup)):
            with open(os.path.join(news_path, newsgroup, news_file), "rb") as rfile:
                text_words = str(rfile.read())
                texts.append(text_words)
                labels.append(label_id)

    return texts, labels, labels_index


def keras_news20_sequence(texts):
    """
    将texts转化为矩阵向量
    :return:
    """
    tokenizer = text.Tokenizer(num_words=20000)
    tokenizer.fit_on_texts(texts)
    word_index = tokenizer.word_index

    sequences = tokenizer.texts_to_sequences(texts)
    datas = sequence.pad_sequences(sequences, maxlen=1000)
    return datas, word_index


def keras_news20_train(news_path):
    """
    训练新闻分类
    :return:
    """
    texts, labels, labels_index = keras_news20_texts(news_path)
    datas, word_index = keras_news20_sequence(texts)

    labels = keras.utils.to_categorical(np.asarray(labels), len(labels_index))

    indices = np.arange(datas.shape[0])
    np.random.shuffle(indices)
    datas = datas[indices]
    labels = labels[indices]

    model = keras.models.Sequential()
    model.add(keras.layers.Embedding(input_dim=len(word_index) + 1, output_dim=128, input_length=1000))

    model.add(keras.layers.Convolution1D(32, 5, activation="relu"))
    model.add(keras.layers.Convolution1D(32, 5, activation="relu"))
    model.add(keras.layers.MaxPooling1D(pool_size=2))

    model.add(keras.layers.Convolution1D(32, 5, activation="relu"))
    model.add(keras.layers.Convolution1D(32, 5, activation="relu"))
    model.add(keras.layers.MaxPooling1D(pool_size=2))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(len(labels_index), activation="softmax"))

    model.summary()

    model.compile(optimizer="adadelta", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(datas, labels, batch_size=64, epochs=10, verbose=1, validation_split=0.2, shuffle=True)


if __name__ == "__main__":
    keras_news20_train(r"D:\Documents\Downloads\news20\20_newsgroup")
