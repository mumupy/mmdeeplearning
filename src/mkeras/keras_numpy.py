#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/12/18 16:07
# @Author  : ganliang
# @File    : keras_numpy.py
# @Desc    : 通过numpy创造数据进行模型训练
import keras
import numpy as np
from keras import Input
from keras.engine import Model
from keras.layers import Dense, LSTM, Embedding, Flatten
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot


def binary_classification():
    """
    逻辑回归
    :return:
    """
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    data = np.random.random((10000, 100))
    labels = np.random.randint(2, size=(10000, 1))

    # Train the model, iterating on the data in batches of 32 samples
    model.fit(data, labels, epochs=10, batch_size=32, verbose=1)


def categorical_classification():
    """
    多值分类
    :return:
    """
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=100))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Generate dummy data

    data = np.random.random((10000, 100))
    labels = np.random.randint(10, size=(10000, 1))

    one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)

    model.fit(data, one_hot_labels, epochs=10, batch_size=32, verbose=1)


def liner_regression():
    """
    线性回归
    :return:
    """

    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=(100,)))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()
    model.compile(optimizer='rmsprop',
                  loss='mse',
                  metrics=['accuracy'])

    data = np.random.random((1000, 100))
    labels = np.mean(data, axis=1)

    model.fit(data, labels, epochs=10, batch_size=32, verbose=1)


def lstm_classification():
    data_dim = 16
    timesteps = 8
    num_classes = 10

    model = Sequential()
    model.add(LSTM(32, return_sequences=True, input_shape=(timesteps, data_dim)))
    model.add(LSTM(32, return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    x_train = np.random.random((1000, timesteps, data_dim))
    y_train = np.random.random((1000, num_classes))

    x_val = np.random.random((100, timesteps, data_dim))
    y_val = np.random.random((100, num_classes))

    model.fit(x_train, y_train, batch_size=64, epochs=5, validation_data=(x_val, y_val))


def stateful_lstm_classification():
    data_dim = 16
    timesteps = 8
    num_classes = 10
    batch_size = 32

    model = Sequential()
    model.add(LSTM(32, return_sequences=True, stateful=True, batch_input_shape=(batch_size, timesteps, data_dim)))
    model.add(LSTM(32, return_sequences=True, stateful=True))
    model.add(LSTM(32, stateful=True))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    x_train = np.random.random((batch_size * 10000, timesteps, data_dim))
    y_train = np.random.random((batch_size * 10000, num_classes))

    x_val = np.random.random((batch_size * 300, timesteps, data_dim))
    y_val = np.random.random((batch_size * 300, num_classes))

    model.fit(x_train, y_train, batch_size=batch_size, epochs=5, shuffle=False, validation_data=(x_val, y_val))


def embeding_result():
    model = Sequential()
    model.add(Embedding(2, 5, input_length=7))  # 输入维，输出维
    model.compile('rmsprop', 'mse')
    result = model.predict(np.array([[0, 1, 0, 1, 1, 0, 0], [1, 1, 0, 0, 1, 0, 1]]))
    print(result)
    print(result.shape)


def embedding_binary_classification():
    docs = ['Well done!',
            'Good work',
            'Great effort',
            'nice work',
            'Excellent!',
            'Weak',
            'Poor effort!',
            'not good',
            'poor work',
            'Could have done better.']

    # define class labels
    labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    vocab_size = 50
    encoded_docs = [one_hot(d, vocab_size) for d in docs]  # one_hot编码到[1,n],不包括0
    print(encoded_docs)

    max_length = 4
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    print(padded_docs)

    input = Input(shape=(4,))
    x = Embedding(vocab_size, 8, input_length=max_length)(input)  # 这一步对应的参数量为50*8
    x = Flatten()(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input, outputs=x)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    model.summary()

    model.fit(padded_docs, labels, epochs=100, verbose=0)
    loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
    print('loss: {0},accuracy:{1}'.format(loss, accuracy))

    loss_test, accuracy_test = model.evaluate(padded_docs, labels, verbose=0)
    print('loss_test: {0},accuracy_test:{1}'.format(loss_test, accuracy_test))

    test = one_hot('Weak', 50)
    padded_test = pad_sequences([test], maxlen=max_length, padding='post')
    print(model.predict(padded_test))


if __name__ == "__main__":
    # binary_classification()
    # categorical_classification()
    # liner_regression()
    # lstm_classification()
    # stateful_lstm_classification()
    # embeding_result()
    embedding_binary_classification()
