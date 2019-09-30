#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/29 16:07
# @Author  : ganliang
# @File    : rnn.py
# @Desc    : 循环神经网络是为更好地处理时序信息而设计的。它引入状态变量来存储过去的信息，并用其与当前的输入共同决定当前的输出。循环
# 神经网络常用于处理序列数据，如一段文字或声音、购物或观影的顺序，甚至是图像中的一行或一列像素。因此，循环神经网络有着极为广泛的实
# 际应用，如语言模型、文本分类、机器翻译、语音识别、图像分析、手写识别和推荐系统。

import d2lzh as d2l
from mxnet import nd


def rnn_jay_lyrics():
    """
    在周杰伦专辑歌词数据集上训练一个模型来进行歌词创作
    :return:
    """

    (corpus_indices, char_to_idx, idx_to_char, vocab_size) = d2l.load_data_jay_lyrics()
    print(nd.one_hot(nd.array([0, 2]), vocab_size))


if __name__ == "__main__":
    rnn_jay_lyrics()
