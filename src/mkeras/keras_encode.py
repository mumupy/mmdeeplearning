#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/12/18 20:23
# @Author  : ganliang
# @File    : keras_encode.py
# @Desc    : keras编码器和解码器
import os

import numpy as np
from keras.datasets import mnist
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from matplotlib import pyplot as plt

from src.config.log import root_path


def show_images(images, decoded_imgs, num_image=10):
    plt.figure(figsize=(20, 4))
    for i in range(1, num_image + 1):
        ax = plt.subplot(2, num_image, i)
        plt.imshow(images[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, num_image, i + num_image)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


def keras_autoencoder():
    encoding_dim = 32

    input_img = Input(shape=(784,))
    encoded = Dense(encoding_dim, activation='relu')(input_img)
    decoded = Dense(784, activation='sigmoid')(encoded)

    autoencoder = Model(inputs=input_img, outputs=decoded)
    encoder = Model(inputs=input_img, outputs=encoded)

    encoded_input = Input(shape=(encoding_dim,))
    decoder_layer = autoencoder.layers[-1]
    decoder = Model(inputs=encoded_input, outputs=decoder_layer(encoded_input))

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=["acc"])

    (x_train, y_train), (x_test, y_test) = mnist.load_data(os.path.join(root_path, "data", "mnist", "mnist.npz"))

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    autoencoder.fit(x_train, x_train, epochs=10, batch_size=256, shuffle=True, validation_data=(x_test, x_test))

    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = decoder.predict(encoded_imgs)

    show_images(x_test, decoded_imgs, 10)


def keras_denses_autoencoder():
    input_img = Input(shape=(784,))
    encoded = Dense(128, activation='relu')(input_img)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(32, activation='relu')(encoded)

    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(784, activation='sigmoid')(decoded)

    autoencoder = Model(inputs=input_img, outputs=decoded)
    autoencoder.summary()
    encoder = Model(inputs=input_img, outputs=encoded)
    encoder.summary()

    encoded_input = Input(shape=(32,))
    encoded_output = autoencoder.layers[-3](encoded_input)
    encoded_output = autoencoder.layers[-2](encoded_output)
    encoded_output = autoencoder.layers[-1](encoded_output)
    decoder = Model(inputs=encoded_input, outputs=encoded_output)
    decoder.summary()

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=["acc"])

    (x_train, y_train), (x_test, y_test) = mnist.load_data(os.path.join(root_path, "data", "mnist", "mnist.npz"))

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    autoencoder.fit(x_train, x_train, epochs=10, batch_size=256, shuffle=True, validation_data=(x_test, x_test))

    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = decoder.predict(encoded_imgs)

    show_images(x_test, decoded_imgs, 10)


def keras_conv_autoencoder():
    """
    卷积自编码
    :return:
    """
    input_img = Input(shape=(28, 28, 1))

    encoded = Convolution2D(128, kernel_size=(3, 3), activation='relu', padding="same")(input_img)
    encoded = MaxPooling2D(pool_size=(2, 2))(encoded)
    encoded = Convolution2D(64, kernel_size=(3, 3), activation='relu', padding="same")(encoded)
    encoded = MaxPooling2D(pool_size=(2, 2))(encoded)
    # encoded = Convolution2D(32, kernel_size=(3, 3), activation='relu', padding="same")(encoded)
    # encoded = MaxPooling2D(pool_size=(2, 2))(encoded)

    # decoded = Convolution2D(32, kernel_size=(3, 3), activation='relu', padding="same")(encoded)
    # decoded = UpSampling2D(size=(2, 2))(decoded)
    decoded = Convolution2D(64, kernel_size=(3, 3), activation='relu', padding="same", name="decoder_layer")(encoded)
    decoded = UpSampling2D(size=(2, 2))(decoded)
    decoded = Convolution2D(128, kernel_size=(3, 3), activation='relu', padding="same")(decoded)
    decoded = UpSampling2D(size=(2, 2))(decoded)

    decoded = Convolution2D(1, kernel_size=(3, 3), activation='sigmoid', padding="same")(decoded)
    autoencoder = Model(inputs=input_img, outputs=decoded)
    autoencoder.summary()
    encoder = Model(inputs=input_img, outputs=encoded)
    encoder.summary()

    decoder_layer = autoencoder.get_layer("decoder_layer")
    encoded_input = Input(shape=decoder_layer.input_shape[1:])
    encoded_output = None
    for layer in autoencoder.layers:
        if layer.name == decoder_layer.name: encoded_output = layer(encoded_input)
        if encoded_output is not None: encoded_output = layer(encoded_output)

    decoder = Model(inputs=encoded_input, outputs=encoded_output)
    decoder.summary()

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=["acc"])

    (x_train, y_train), (x_test, y_test) = mnist.load_data(os.path.join(root_path, "data", "mnist", "mnist.npz"))

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, x_train.shape + (1,))
    x_test = np.reshape(x_test, x_test.shape + (1,))

    autoencoder.fit(x_train, x_train, epochs=10, batch_size=256, shuffle=True, validation_data=(x_test, x_test))

    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = decoder.predict(encoded_imgs)

    show_images(x_test, decoded_imgs, 10)


def keras_noiceconv_autoencoder():
    """
    图片去燥自编码器
    :return:
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data(os.path.join(root_path, "data", "mnist", "mnist.npz"))

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, x_train.shape + (1,))
    x_test = np.reshape(x_test, x_test.shape + (1,))

    noise_factor = 0.5
    x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
    x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    x_test_noisy = np.clip(x_test_noisy, 0., 1.)

    show_images(x_train, x_train_noisy, 10)

    kernel_size = (3, 3)
    pool_size = (2, 2)

    input_data = Input(shape=(28, 28, 1))
    encoded = Convolution2D(128, kernel_size=kernel_size, activation="relu", padding="same")(input_data)
    encoded = MaxPooling2D(pool_size=pool_size)(encoded)
    encoded = Convolution2D(64, kernel_size=kernel_size, activation="relu", padding="same")(encoded)
    encoded = MaxPooling2D(pool_size=pool_size)(encoded)

    decoded = Convolution2D(64, kernel_size=kernel_size, activation="relu", padding="same")(encoded)
    decoded = UpSampling2D(size=pool_size)(decoded)
    decoded = Convolution2D(128, kernel_size=kernel_size, activation="relu", padding="same")(decoded)
    decoded = UpSampling2D(size=pool_size)(decoded)
    decoded = Convolution2D(1, kernel_size=kernel_size, activation="sigmoid", padding="same")(decoded)

    autoencoder = Model(inputs=input_data, outputs=decoded)
    autoencoder.summary()

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=["acc"])
    autoencoder.fit(x_train_noisy, x_train, batch_size=128, epochs=10, verbose=1,
                    validation_data=(x_test_noisy, x_test))

    decode_images = autoencoder.predict(x_test_noisy, batch_size=128)
    show_images(x_test_noisy, decode_images, 10)


if __name__ == "__main__":
    # keras_autoencoder()
    # keras_denses_autoencoder()
    # keras_conv_autoencoder()
    keras_noiceconv_autoencoder()
