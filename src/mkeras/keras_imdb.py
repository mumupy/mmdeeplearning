#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/27 14:58
# @Author  : ganliang
# @File    : keras_imdb.py
# @Desc    : imdb电影评价,评价为正面1或者反面0
import os

import keras
import numpy as np
from keras.preprocessing import sequence, text

from mkeras import keras_history_plotcurve

imdb = keras.datasets.imdb
from src.config import logger, root_path


def keras_imdb_info(num_words=None, maxlen=None):
    (x_train, y_train), (x_test, y_test) = imdb.load_data(
        os.path.join(root_path, "data", "imdb", "imdb.npz"), num_words=num_words)

    if not num_words: num_words = max(max([max(x) for x in x_train]), max([max(x) for x in x_test])) + 1
    if not maxlen: maxlen = max(max([len(x) for x in x_train]), max([len(x) for x in x_test])) + 1

    logger.info("train data shape {0}".format(x_train.shape))
    logger.info("train labels shape {0}".format(y_train.shape))
    logger.info("test data shape {0}".format(x_test.shape))
    logger.info("test label shape {0}".format(y_test.shape))

    logger.info("num_words {0}".format(num_words))
    logger.info("maxlen {0}".format(maxlen))

    # 一个映射单词到整数索引的词典
    word_index = imdb.get_word_index(os.path.join(root_path, "data", "imdb", "imdb_word_index.json"))
    logger.info(len(word_index))


def keras_imdb_mlp(num_words=None, maxlen=None):
    """
    imdb分类，构建多层感知机进行分类
    :param num_words:
    :param maxlen:
    :return:
    """
    (x_train, y_train), (x_test, y_test) = imdb.load_data(
        os.path.join(root_path, "data", "imdb", "imdb.npz"), num_words=num_words)

    if not num_words: num_words = max(max([max(x) for x in x_train]), max([max(x) for x in x_test])) + 1
    if not maxlen: maxlen = max(max([len(x) for x in x_train]), max([len(x) for x in x_test])) + 1

    tokenizer = text.Tokenizer(num_words=num_words)
    x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
    x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')

    input = keras.layers.Input(shape=(num_words,))
    x = keras.layers.Dense(512, activation="relu", kernel_initializer="glorot_normal")(input)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.25)(x)

    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.25)(x)

    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.25)(x)

    x = keras.layers.Dense(1, activation="sigmoid")(x)

    model = keras.models.Model(inputs=input, outputs=x)
    model.summary()

    model.compile(optimizer="adadelta", loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2, verbose=1)
    keras_history_plotcurve(history)

    # 评估模型
    test_loss, test_accuracy = model.evaluate(x_test, y_test, batch_size=64, verbose=0)
    logger.info("\ntest_loss:{0},test_accuracy:{1}".format(test_loss, test_accuracy))


def keras_imdb_cnn(num_words=None, maxlen=None, embedding_dim=128):
    """
    imdb的cnn评分
    :return:
    """
    (x_train, y_train), (x_test, y_test) = imdb.load_data(
        os.path.join(root_path, "data", "imdb", "imdb.npz"), num_words=num_words)

    if not num_words: num_words = max(max([max(x) for x in x_train]), max([max(x) for x in x_test])) + 1
    if not maxlen: maxlen = max(max([len(x) for x in x_train]), max([len(x) for x in x_test])) + 1

    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

    model = keras.models.Sequential()
    model.add(keras.layers.Embedding(num_words + 1, embedding_dim, input_length=maxlen))

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

    history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2, verbose=1)
    keras_history_plotcurve(history)
    # 评估模型
    test_loss, test_accuracy = model.evaluate(x_test, y_test, batch_size=64, verbose=0)
    logger.info("\ntest_loss:{0},test_accuracy:{1}".format(test_loss, test_accuracy))


def keras_imdb_rnn(num_words=None, maxlen=None, embedding_dim=128):
    """
    imdb影评二分类，使用simpleRNN进行分类
    :return:
    """
    (x_train, y_train), (x_test, y_test) = imdb.load_data(
        os.path.join(root_path, "data", "imdb", "imdb.npz"), num_words=num_words)

    if not num_words: num_words = max(max([max(x) for x in x_train]), max([max(x) for x in x_test])) + 1
    if not maxlen: maxlen = max(max([len(x) for x in x_train]), max([len(x) for x in x_test])) + 1

    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

    model = keras.models.Sequential()
    model.add(keras.layers.Embedding(num_words + 1, embedding_dim, input_length=maxlen))
    model.add(keras.layers.SimpleRNN(128))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.summary()
    model.compile(optimizer="rmsprop", loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2, verbose=1)
    keras_history_plotcurve(history)
    # 评估模型
    test_loss, test_accuracy = model.evaluate(x_test, y_test, batch_size=64, verbose=0)
    logger.info("\ntest_loss:{0},test_accuracy:{1}".format(test_loss, test_accuracy))


def keras_imdb_lstm(num_words=None, maxlen=None, embedding_dim=128):
    """
    imdb影评二分类，使用lstm长短期记忆进行分类
    :return:
    """
    (x_train, y_train), (x_test, y_test) = imdb.load_data(
        os.path.join(root_path, "data", "imdb", "imdb.npz"), num_words=num_words)

    if not num_words: num_words = max(max([max(x) for x in x_train]), max([max(x) for x in x_test])) + 1
    if not maxlen: maxlen = max(max([len(x) for x in x_train]), max([len(x) for x in x_test])) + 1

    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

    model = keras.models.Sequential()
    model.add(keras.layers.Embedding(num_words + 1, embedding_dim, input_length=maxlen))
    model.add(keras.layers.LSTM(128))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.summary()
    model.compile(optimizer="rmsprop", loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2, verbose=1)
    keras_history_plotcurve(history)
    # 评估模型
    test_loss, test_accuracy = model.evaluate(x_test, y_test, batch_size=64, verbose=0)
    logger.info("\ntest_loss:{0},test_accuracy:{1}".format(test_loss, test_accuracy))


def keras_imdb_gru(num_words=None, maxlen=None, embedding_dim=128):
    """
    imdb影评二分类，使用gru门控循环单元进行分类
    :return:
    """
    (x_train, y_train), (x_test, y_test) = imdb.load_data(
        os.path.join(root_path, "data", "imdb", "imdb.npz"), num_words=num_words)

    if not num_words: num_words = max(max([max(x) for x in x_train]), max([max(x) for x in x_test])) + 1
    if not maxlen: maxlen = max(max([len(x) for x in x_train]), max([len(x) for x in x_test])) + 1

    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

    model = keras.models.Sequential()
    model.add(keras.layers.Embedding(num_words + 1, embedding_dim, input_length=maxlen))
    model.add(keras.layers.GRU(128))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.summary()
    model.compile(optimizer="rmsprop", loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2, verbose=1)
    keras_history_plotcurve(history)

    # 评估模型
    test_loss, test_accuracy = model.evaluate(x_test, y_test, batch_size=64, verbose=0)
    logger.info("\ntest_loss:{0},test_accuracy:{1}".format(test_loss, test_accuracy))


def keras_imdb_cnn_lstm(num_words=None, maxlen=None, embedding_dim=128):
    """
    imdb二分类，使用cnn加上lstm进行分类
    :return:
    """
    (x_train, y_train), (x_test, y_test) = imdb.load_data(
        os.path.join(root_path, "data", "imdb", "imdb.npz"), num_words=num_words)

    if not num_words: num_words = max(max([max(x) for x in x_train]), max([max(x) for x in x_test])) + 1
    if not maxlen: maxlen = max(max([len(x) for x in x_train]), max([len(x) for x in x_test])) + 1

    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

    model = keras.models.Sequential()
    model.add(keras.layers.Embedding(num_words + 1, embedding_dim, input_length=maxlen))

    model.add(keras.layers.Conv1D(32, 5, activation="relu"))
    model.add(keras.layers.MaxPool1D())
    model.add(keras.layers.Dropout(0.4))

    model.add(keras.layers.Conv1D(32, 5, activation="relu"))
    model.add(keras.layers.MaxPool1D())
    model.add(keras.layers.Dropout(0.4))

    model.add(keras.layers.LSTM(64))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.summary()
    model.compile(optimizer="rmsprop", loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2, verbose=1)
    keras_history_plotcurve(history)

    # 评估模型
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    logger.info("\ntest_loss:{0},test_accuracy:{1}".format(test_loss, test_accuracy))


def keras_imdb_fasttext(num_words=None, maxlen=None, batch_size=32, embedding_dims=50, epochs=5, ngram_range=1):
    """
    ngram 训练
    :param num_words:  词典单词数量
    :param maxlen:  句子长度
    :param batch_size:  批量大下
    :param embedding_dims:  嵌入层输出层数
    :param epochs:  训练轮数
    :param ngram_range:  ngram数量
    :return:
    """

    def create_ngram_set(input_list, ngram_value=2):
        """
        Extract a set of n-grams from a list of integers.
        >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
        {(4, 9), (4, 1), (1, 4), (9, 4)}
        >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
        [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
        """
        return set(zip(*[input_list[i:] for i in range(ngram_value)]))

    def add_ngram(sequences, token_indice, ngram_range=2):
        """
        Augment the input list of list (sequences) by appending n-grams values.
        Example: adding bi-gram
        >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
        >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
        >>> add_ngram(sequences, token_indice, ngram_range=2)
        [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]
        Example: adding tri-gram
        >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
        >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017, (7, 9, 2): 2018}
        >>> add_ngram(sequences, token_indice, ngram_range=3)
        [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42, 2018]]
        """
        new_sequences = []
        for input_list in sequences:
            new_list = input_list[:]
            for ngram_value in range(2, ngram_range + 1):
                for i in range(len(new_list) - ngram_value + 1):
                    ngram = tuple(new_list[i:i + ngram_value])
                    if ngram in token_indice:
                        new_list.append(token_indice[ngram])
            new_sequences.append(new_list)

        return new_sequences

    print('Loading data...')
    (x_train, y_train), (x_test, y_test) = imdb.load_data(
        os.path.join(root_path, "data", "imdb", "imdb.npz"), num_words=num_words)
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')
    print('Average train sequence length: {}'.format(np.mean(list(map(len, x_train)), dtype=int)))
    print('Average test sequence length: {}'.format(np.mean(list(map(len, x_test)), dtype=int)))

    if not num_words: num_words = max(max([max(x) for x in x_train]), max([max(x) for x in x_test])) + 1
    if not maxlen: maxlen = max(max([len(x) for x in x_train]), max([len(x) for x in x_test])) + 1

    if ngram_range > 1:
        print('Adding {}-gram features'.format(ngram_range))
        ngram_set = set()
        for input_list in x_train:
            for i in range(2, ngram_range + 1):
                set_of_ngram = create_ngram_set(input_list, ngram_value=i)
                ngram_set.update(set_of_ngram)

        # Dictionary mapping n-gram token to a unique integer.
        # Integer values are greater than max_features in order
        # to avoid collision with existing features.
        start_index = num_words + 1
        token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
        indice_token = {token_indice[k]: k for k in token_indice}

        # max_features is the highest integer that could be found in the dataset.
        num_words = np.max(list(indice_token.keys())) + 1

        # Augmenting x_train and x_test with n-grams features
        x_train = add_ngram(x_train, token_indice, ngram_range)
        x_test = add_ngram(x_test, token_indice, ngram_range)
        print('Average train sequence length: {}'.format(np.mean(list(map(len, x_train)), dtype=int)))
        print('Average test sequence length: {}'.format(np.mean(list(map(len, x_test)), dtype=int)))

    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    print('Build model...')
    model = keras.models.Sequential()

    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    model.add(keras.layers.Embedding(num_words, embedding_dims, input_length=maxlen))

    # we add a GlobalAveragePooling1D, which will average the embeddings
    # of all words in the document
    model.add(keras.layers.GlobalAveragePooling1D())

    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.summary()

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))
    keras_history_plotcurve(history)


if __name__ == "__main__":
    # keras_imdb_info()
    # keras_imdb_mlp()
    # keras_imdb_cnn()
    # keras_imdb_rnn()
    # keras_imdb_lstm()
    # keras_imdb_gru()
    keras_imdb_cnn_lstm(num_words=20000, maxlen=400)
    # keras_imdb_fasttext(num_words=20000, maxlen=400, ngram_range=2)
