#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/8 17:09
# @Author  : ganliang
# @File    : word2vec.py
# @Desc    : word2vec


from mxnet import nd
from mxnet.contrib import text

from src.config import logger


def knn(W, x, k):
    # 添加的1e-9是为了数值稳定性
    cos = nd.dot(W, x.reshape((-1,))) / (
            (nd.sum(W * W, axis=1) + 1e-9).sqrt() * nd.sum(x * x).sqrt())
    topk = nd.topk(cos, k=k, ret_typ='indices').asnumpy().astype('int32')
    return topk, [cos[i].asscalar() for i in topk]


def get_similar_tokens(query_token, k, embed):
    """
    求近义词
    :param query_token: 去查单词
    :param k: 返回多少相似单词
    :param embed: 词典
    :return:
    """
    topk, cos = knn(embed.idx_to_vec,
                    embed.get_vecs_by_tokens([query_token]), k + 1)
    for i, c in zip(topk[1:], cos[1:]):  # 除去输入词
        logger.info('cosine sim=%.3f: %s' % (c, (embed.idx_to_token[i])))


def get_analogy(tokens, embed):
    """
    求类比词
    :param tokens: 词汇
    :param embed: 词典
    :return:
    """
    vecs = embed.get_vecs_by_tokens(tokens)
    x = vecs[1] - vecs[0] + vecs[2]
    topk, cos = knn(embed.idx_to_vec, x, 1)
    analogy = embed.idx_to_token[topk[0]]
    logger.info(analogy)
    return analogy


if __name__ == "__main__":
    logger.info(text.embedding.get_pretrained_file_names().keys())
    logger.info(text.embedding.get_pretrained_file_names('glove'))
    logger.info(text.embedding.get_pretrained_file_names('fasttext'))

    # glove_6b50d = text.embedding.create('glove', pretrained_file_name='glove.6B.50d.txt')
    glove_6b50d = text.embedding.create('fasttext', pretrained_file_name='wiki.zh.vec')
    get_similar_tokens('chip', 3, glove_6b50d)
    get_similar_tokens('deeplearning', 3, glove_6b50d)

    # get_analogy(['man', 'woman', 'son', "wife"], glove_6b50d)
    # get_analogy(['beijing', 'china', 'tokyo'], glove_6b50d)
    # get_analogy(['bad', 'worst', 'big'], glove_6b50d)
    # get_analogy(['do', 'did', 'go'], glove_6b50d)
    # get_analogy(['北京', '中国', '东京'], glove_6b50d)
    get_analogy(['郑州', '河南', '武汉'], glove_6b50d)
