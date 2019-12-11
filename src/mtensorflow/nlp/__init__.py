#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/12/11 17:23
# @Author  : ganliang
# @File    : __init__.py.py
# @Desc    : nlp人工只能
import numpy
import speech_data
import tflearn


def speech2text():
    """
    英文数字语音识别
    :return:
    """
    learning_rate = 0.0001
    training_iters = 300000  # 迭代次数
    batch_size = 64

    width = 20  # MFCC特征
    height = 80  # 最大发音长度
    classes = 10  # 数字类别

    batch = speech_data.mfcc_batch_generator(batch_size)

    X, Y = next(batch)
    trainX, trainY = X, Y
    testX, testY = X, Y

    net = tflearn.input_data([None, width, height])
    net = tflearn.lstm(net, 128, dropout=0.8)
    net = tflearn.fully_connected(net, classes, activation='softmax')
    net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy')

    model = tflearn.DNN(net, tensorboard_verbose=0)

    model.fit(trainX, trainY, n_epoch=10, validation_set=(testX, testY), show_metric=True, batch_size=batch_size)
    _y = model.predict(X)
    model.save("tflearn.lstm.model")

    demo_file = "5_Vicki_260.wav"
    demo = speech_data.load_wav_file(speech_data.path + demo_file)
    result = model.predict([demo])
    result = numpy.argmax(result)
    print("predicted digit for %s : result = %d " % (demo_file, result))
