#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/12/13 17:14
# @Author  : ganliang
# @File    : keras_skipgram.py
# @Desc    : skipgrams
from PIL import Image, ImageFilter
from keras.preprocessing import sequence, text, image


def test_sequence_skipgrams():
    couples, labels = sequence.skipgrams([0, 1, 2, 3], vocabulary_size=4, window_size=2)
    print(couples)
    print(labels)


def test_sequence_make_sampling_table():
    print(sequence.make_sampling_table(5))


def test_text_to_word_sequence():
    seq = text.text_to_word_sequence("i love you ? babymm qianqian babymm qq!", lower=True, split=" ")
    print(seq)


def test_text_one_hot():
    print(text.one_hot("i love you", 100))


def test_text_tokenizer():
    texts = ["i love you babymumu", "babymm is my son", "babyqq is my son", "cws is my wife"]
    tokenizer = text.Tokenizer(num_words=10)
    tokenizer.fit_on_texts(texts)
    print(tokenizer.word_counts)
    print(tokenizer.word_docs)
    print(tokenizer.word_index)

    for seq in tokenizer.texts_to_sequences(texts):
        print(seq)


def test_image_img_to_array():
    im = Image.open(r"E:/小柚子/1.jpg")
    imdatas = image.img_to_array(im, data_format="channels_last")
    print(imdatas)
    image.array_to_img(imdatas, data_format="channels_last")

    im2 = im.filter(ImageFilter.GaussianBlur)
    im2.save('E:/小柚子/blur.jpg', 'jpeg')


if __name__ == "__main__":
    # test_text_one_hot()
    # test_text_tokenizer()
    test_image_img_to_array()
