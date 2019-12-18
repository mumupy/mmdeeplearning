#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/12/13 17:14
# @Author  : ganliang
# @File    : keras_skipgram.py
# @Desc    : skipgrams
import os

import numpy as np
from PIL import Image, ImageFilter
from keras.preprocessing import sequence, text, image


def sequence_skipgrams():
    couples, labels = sequence.skipgrams([0, 1, 2, 3], vocabulary_size=4, window_size=2)
    print(couples)
    print(labels)


def sequence_padsequences():
    texts = ["i love you ? babymm qianqian babymm qq!", "babymm lovecws", "ceshi qianqian"]
    tokenizer = text.Tokenizer(num_words=100)
    tokenizer.fit_on_texts(texts)
    print(tokenizer.word_index)
    print(tokenizer.word_docs)
    print(tokenizer.word_counts)

    sequences = tokenizer.texts_to_sequences(texts)
    print(sequence.pad_sequences(sequences, maxlen=10))


def sequence_make_sampling_table():
    print(sequence.make_sampling_table(5))


def text_to_word_sequence():
    seq = text.text_to_word_sequence("i love you ? babymm qianqian babymm qq!", lower=True, split=" ")
    print(seq)


def text_one_hot():
    print(text.one_hot("i love you", 50))


def text_tokenizer():
    texts = ["i love you babymumu", "babymm is my son", "babyqq is my son", "cws is my wife"]
    tokenizer = text.Tokenizer(num_words=10)
    tokenizer.fit_on_texts(texts)
    print(tokenizer.word_counts)
    print(tokenizer.word_docs)
    print(tokenizer.word_index)

    for seq in tokenizer.texts_to_sequences(texts): print(seq)


def image_img_to_array():
    im = Image.open(r"E:/小柚子/1.jpg")
    imdatas = image.img_to_array(im, data_format="channels_last")
    print(imdatas)
    image.array_to_img(imdatas, data_format="channels_last")

    im2 = im.filter(ImageFilter.GaussianBlur)
    im2.save('E:/小柚子/blur.jpg', 'jpeg')


def images_datagen():
    from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img
    os.environ[''] = "theano"

    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    img = load_img(r"E:/小柚子/1.jpg", target_size=(250, 250))
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)

    i = 0
    for batch in datagen.flow(x, batch_size=1, save_to_dir=r'E:\小柚子\preview', save_prefix='youzi', save_format='jpeg'):
        if batch.ndim == 4: batch = np.reshape(batch, batch.shape[1:])
        i += 1
        array_to_img(batch)
        if i >= 20: break


if __name__ == "__main__":
    # sequence_padsequences()
    # text_to_word_sequence()
    text_one_hot()
    # text_tokenizer()
    # image_img_to_array()
    # images_datagen()
