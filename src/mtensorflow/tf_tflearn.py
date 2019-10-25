#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/25 13:55
# @Author  : ganliang
# @File    : test_tf_tflearn.py
# @Desc    : tflearn测试

import tensorflow as tf
import tflearn
from tflearn import DNN
from tflearn.data_augmentation import ImageAugmentation
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_utils import shuffle, to_categorical, pad_sequences
from tflearn.datasets import cifar10, imdb
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


def tflearn_cifar():
    """
    图像分类
    :return:
    """

    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    X_train, Y_train = shuffle(X_train, Y_train)
    Y_train = to_categorical(Y_train, nb_classes=10)
    Y_test = to_categorical(Y_test, nb_classes=10)

    # 对数据集进行零中心化（即对整个数据集计算平均值），同时进行 STD 标准化（即对整个数据集计算标准差）
    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()

    # 通过随机左右翻转和随机旋转来增强数据集
    img_aug = ImageAugmentation()
    img_aug.add_random_flip_leftright()
    img_aug.add_random_rotation(max_angle=25.)

    # 定义模型
    network = input_data(shape=(None, 32, 32, 3), data_preprocessing=img_prep, data_augmentation=img_aug)
    network = conv_2d(network, 32, 3, activation="relu")
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 64, 3, activation="relu")
    network = conv_2d(network, 64, 3, activation="relu")
    network = max_pool_2d(network, 2)
    network = fully_connected(network, 512, activation="relu")
    network = dropout(network, 0.5)
    network = fully_connected(network, 10, activation="softmax")
    network = regression(network, optimizer="adam", loss="categorical_crossentropy", learning_rate=0.001)

    # 训练模型
    model = DNN(network, tensorboard_verbose=0)
    model.fit(X_train, Y_train, n_epoch=50, shuffle=True, validation_set=(X_test, Y_test), show_metric=True,
              batch_size=96, run_id="cifar10_cnn")


def tflearn_imdb():
    """
    文本情感分析
    :return:
    """
    (X_train, Y_train), (X_test, Y_test) = imdb.load_data()

    X_train, Y_train = pad_sequences(Y_train, maxlen=100), to_categorical(Y_train, nb_classes=2)
    X_test, Y_test = pad_sequences(Y_test, maxlen=100), to_categorical(Y_test, nb_classes=2)

    network = input_data([None, 100], name="input")
    tflearn.embedding(network, input_dim=10000, output_dim=128)

    branch1 = tflearn.conv_1d(network, 128, 3, padding="valid", activation="relu", regularizer="L2")
    branch2 = tflearn.conv_1d(network, 128, 4, padding="valid", activation="relu", regularizer="L2")
    branch3 = tflearn.conv_1d(network, 128, 5, padding="valid", activation="relu", regularizer="L2")

    network = tflearn.merge([branch1, branch2, branch3], mode="concat", axis=1)
    network = tf.expand_dims(network, 2)
    network = tflearn.global_avg_pool(network)
    network = tflearn.dropout(network, 0.5)
    network = tflearn.fully_connected(network, 2, activation="softmax")

    network = tflearn.regression(network, optimizer="adam", learning_rate=0.001, loss="categorical_crossentropy",
                                 name="target")

    model = tflearn.DNN(network, tensorboard_verbose=0)
    model.fit(X_train, Y_train, n_epoch=5, shuffle=True, validation_set=(X_test, Y_test), show_metric=True,
              batch_size=32)
