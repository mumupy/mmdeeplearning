#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/24 18:28
# @Author  : ganliang
# @File    : tf_keras.py
# @Desc    : keras高级api

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.layers import Dense
from keras.losses import mean_squared_error
from keras.models import Sequential
from sklearn import datasets
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.config import logger


def keras_boston(epochs=10, learning_rate=0.001, batch_size=200, hidden=30):
    """
    使用tensorflow的高级api执行房价预估
    :return:
    """
    boston_data = datasets.load_boston()

    X_train, X_test, y_train, y_test = train_test_split(boston_data.data, boston_data.target, test_size=0.3,
                                                        random_state=42)
    # 数据归一化
    minmax_scaler = MinMaxScaler()
    X_train = minmax_scaler.fit_transform(X_train)
    X_test = minmax_scaler.fit_transform(X_test)
    y_train = minmax_scaler.fit_transform(np.reshape(y_train, newshape=(-1, 1)))
    y_test = minmax_scaler.fit_transform(np.reshape(y_test, newshape=(-1, 1)))

    n_samples, n_feature = X_train.shape

    model = Sequential()
    model.add(Dense(hidden, input_dim=n_feature, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.summary()

    model.compile(optimizer="adam", loss="mean_squared_error", )
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=1)
    y_test_pred = model.predict(X_test)
    # y_train_pred = model.predict(X_train)
    r2 = r2_score(y_test, y_test_pred)
    rmse = mean_squared_error(np.reshape(y_test, (-1,)), np.reshape(y_test_pred, (-1,)))
    with tf.Session(): logger.info("test rs {0}  rmse {1}".format(r2, rmse.eval()))

    y_test_pred = minmax_scaler.inverse_transform(y_test_pred)
    y_test = minmax_scaler.inverse_transform(y_test)

    # data = pd.DataFrame(np.c_[y_test_pred, y_test], columns=["epoch", "price"])
    # sns.lineplot(x="epoch", y="price", data=data)
    # plt.show()

    plt.plot(list(range(len(y_test_pred))), y_test_pred, "ro", label="test pred")
    plt.plot(list(range(len(y_test))), y_test, "bo", label="test real")
    plt.xlabel("epoch")
    plt.xlabel("price")
    plt.title("test pred/real")
    plt.legend()
    plt.show()


def super_variables(epochs=10, min_loss=0.01, batch_size=200, hidden=30):
    """
    超参数
    :return:
    """

    boston_data = datasets.load_boston()
    X_train, X_test, y_train, y_test = train_test_split(boston_data.data, boston_data.target, test_size=0.3,
                                                        random_state=42)
    # 数据归一化
    minmax_scaler = MinMaxScaler()
    X_train = minmax_scaler.fit_transform(X_train)
    X_test = minmax_scaler.fit_transform(X_test)
    y_train = minmax_scaler.fit_transform(np.reshape(y_train, newshape=(-1, 1)))
    y_test = minmax_scaler.fit_transform(np.reshape(y_test, newshape=(-1, 1)))

    n_samples, n_feature = X_train.shape

    model = Sequential()
    model.add(Dense(hidden, input_dim=n_feature, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.summary()
    model_json = "{}"
    with tf.Session() as sess:
        # saver = tf.train.Saver()
        # save_path = saver.save(sess, "model.ckpt")
        # logger.info("Model saved in %s" % save_path)
        for epoch in range(epochs):
            batch_count = len(X_train) // batch_size
            for i in range(batch_count):
                model.compile(optimizer="adam", loss="mean_squared_error", )
                model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size,
                          verbose=1)
                y_test_pred = model.predict(X_test)
                rmse = mean_squared_error(np.reshape(y_test, (-1,)), np.reshape(y_test_pred, (-1,))).eval()
                logger.info(rmse)
                if rmse < min_loss:
                    min_loss = rmse
                    model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
            model.save_weights("model.hdf5")
            logger.info("Saved model to disk")


def vis_visual():
    """
    vis可视化卷积层
    :return:
    """


def image_classify():
    """
    图像分类
    :return:
    """
    import keras.applications as applications
    from keras.preprocessing.image import img_to_array, load_img
    from keras.applications import imagenet_utils
    from PIL import Image

    MODELS = {
        "vgg16": (applications.VGG16, (224, 224)),
        "vgg19": (applications.VGG19, (244, 244)),
        "inception": (applications.InceptionV3, (299, 299)),
        "xception": (applications.Xception, (299, 299)),
        "resnet": (applications.ResNet50, (224, 224))
    }

    def image_load_and_convert(image_path, model):
        pil_im = Image.open(image_path, "r")
        plt.imshow(pil_im)
        image = img_to_array(load_img(image_path, target_size=MODELS[model][1]))
        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)
        return image

    def classify_image(image_path, model):
        img = image_load_and_convert(image_path, model)
        Network = MODELS[model][0]
        model = Network(weights="imagenet")
        predicts = model.predict(img)
        p = imagenet_utils.decode_predictions(predicts)
        for i, (imagenetId, label, prob) in enumerate(p[0]):
            logger.info("{0} : {1}: {2}".format(i, label, prob * 100))

    def print_model(model):
        logger.info("Model:{0}".format(model))
        Network = MODELS[model][0]
        model = Network(weights="imagenet")
        logger.info("\n%s" % model.summary())

    print_model("vgg19")
