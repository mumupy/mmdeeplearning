#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/12/16 17:00
# @Author  : ganliang
# @File    : keras_glove.py
# @Desc    : glove预训练词向量
import keras
import numpy as np
from keras.preprocessing import text, sequence


def keras_glove():
    """
    预测新闻类型 监督性任务
    :return:
    """
    # 每个文本保留的最大单词数量
    MAX_SEQUENCE_LENGTH = 1000
    # 最多的单词数量
    MAX_NB_WORDS = 20000
    # 每个单词生成100维的向量
    EMBEDDING_DIM = 100

    # 读取新闻数据
    texts = ["lovecws", "babymm", "babyqq", "cws", "mumu"]
    labels = [0, 1, 1, 2, 3]

    tokenizer = text.Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)

    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index

    data = sequence.pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    labels = keras.utils.to_categorical(np.asarray(labels))

    # 数据打乱
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)

    data = data[indices]
    labels = labels[indices]

    sequence_input = keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,), dtype="int32")

    embedding_layer = keras.layers.Embedding(len(word_index) + 1, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH)
    embedding_sequence = embedding_layer(sequence_input)

    x = keras.layers.Conv1D(256, 5, activation="relu")(embedding_sequence)
    x = keras.layers.MaxPooling1D(5)(x)
    x = keras.layers.Dropout(0.4)(x)

    x = keras.layers.Conv1D(256, 5, activation="relu")(x)
    x = keras.layers.MaxPooling1D(5)(x)
    x = keras.layers.Dropout(0.4)(x)

    x = keras.layers.Conv1D(256, 5, activation="relu")(x)
    x = keras.layers.MaxPooling1D(5)(x)
    x = keras.layers.Dropout(0.4)(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    preds = keras.layers.Dense(4, activation="softmax")(x)

    model = keras.models.Model(sequence_input, preds)
    model.summary()

    model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

    model.fit(data, labels, batch_size=64, epochs=10, validation_split=0.2)


if __name__ == "__main__":
    keras_glove()
