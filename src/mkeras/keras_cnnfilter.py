#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/12/19 19:43
# @Author  : ganliang
# @File    : keras_cnnfilter.py
# @Desc    : cnn滤波器

import keras
import numpy as np
from keras import backend as K
from scipy.misc import imsave


def keras_cnnfilter():
    """
    cnn查看滤波器
    :return:
    """
    model = keras.applications.VGG16(include_top=False, weights='imagenet')
    first_layer = model.layers[-1]
    input_img = first_layer.input

    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    layer_name = 'block5_conv3'
    filter_index = 0
    img_width, img_height = 128, 128

    layer_output = layer_dict[layer_name].output
    loss = K.mean(layer_output[:, :, :, filter_index])

    grads = K.gradients(loss, input_img)[0]

    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    iterate = K.function([input_img], [loss, grads])

    input_img_data = np.random.random((1, 3, img_width, img_height)) * 20 + 128.
    for i in range(20):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * i

    def deprocess_image(x):
        x -= x.mean()
        x /= (x.std() + 1e-5)
        x *= 0.1

        x += 0.5
        x = np.clip(x, 0, 1)

        x *= 255
        x = x.transpose((1, 2, 0))
        x = np.clip(x, 0, 255).astype('uint8')
        return x

    img = input_img_data[0]
    img = deprocess_image(img)
    imsave('%s_filter_%d.png' % (layer_name, filter_index), img)


if __name__ == "__main__":
    keras_cnnfilter()
