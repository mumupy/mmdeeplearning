#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/12/17 19:26
# @Author  : ganliang
# @File    : keras_wrappers.py
# @Desc    : keras闭包器
from keras.wrappers import scikit_learn


def keras_wrappers():
    keras_classifier = scikit_learn.KerasClassifier()
    keras_classifier.predict()
