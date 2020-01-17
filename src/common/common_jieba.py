#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/1/16 9:18
# @Author  : ganliang
# @File    : common_jieba.py
# @Desc    : tfidf特征提取
import os

import jieba
import jieba.analyse
import jieba.posseg

from src.config.log import logger, root_path


def cut():
    cuts = jieba.cut("周大福是创新办主任也是云计算方面的专家", cut_all=False)
    logger.info(",".join(cuts))


def cut_for_search():
    cuts = jieba.cut_for_search("周大福是创新办主任也是云计算方面的专家")
    logger.info(",".join(cuts))


def load_userdict():
    """
    添加用户自定义词典
    :return:
    """
    jieba.load_userdict(os.path.join(root_path, "data", "jieba", "userdict.txt"))
    cuts = jieba.cut("周大福是创新办主任也是云计算方面的专家", cut_all=False)
    logger.info(",".join(cuts))


def dictionary():
    """
    使用自定义词典
    :return:
    """
    jieba.set_dictionary(os.path.join(root_path, "data", "jieba", "dict.txt.big.txt"))
    cuts = jieba.cut("周大福是创新办主任也是云计算方面的专家", cut_all=False)
    logger.info(",".join(cuts))


def posseg():
    """
    词性标注
    :return:
    """
    words = jieba.posseg.cut("我爱北京天安门")
    for word, flag in words: logger.info('%s %s' % (word, flag))


def add_word():
    jieba.add_word("创新办")
    jieba.add_word("专家")
    jieba.add_word("云计算")
    jieba.del_word("大福")
    cuts = jieba.cut("周大福是创新办主任也是云计算方面的专家", cut_all=False)
    logger.info(",".join(cuts))


def stop_words():
    s = "周大福是创新办主任也是云计算方面的专家"
    jieba.analyse.set_stop_words(os.path.join(root_path, "data", "jieba", "stopwords.txt"))
    tags = jieba.analyse.extract_tags(s, topK=5, withWeight=True)
    for x, w in tags: logger.info('%s %s' % (x, w))


def tf_idf():
    """
    tfidf关键词提取   tf=词w在文档d出现的次数/文档d包含的词数量，idf=log(总文档数量D/(包含词w的文档数量+1))
    :return:
    """
    s = "此外，公司拟对全资子公司吉林欧亚置业有限公司增资4.3亿元，增资后，吉林欧亚置业注册资本由7000万元增加到5亿元。吉林欧亚置业主要经营范围为房地产开发及百货零售等业务。目前在建吉林欧亚城市商业综合体项目。2013年，实现营业收入0万元，实现净利润-139.13万元。"
    tags = jieba.analyse.extract_tags(s, topK=20, withWeight=True)
    for x, w in tags: logger.info('%s %s' % (x, w))


def custorm_idf():
    s = "此外，公司拟对全资子公司吉林欧亚置业有限公司增资4.3亿元，增资后，吉林欧亚置业注册资本由7000万元增加到5亿元。吉林欧亚置业主要经营范围为房地产开发及百货零售等业务。目前在建吉林欧亚城市商业综合体项目。2013年，实现营业收入0万元，实现净利润-139.13万元。"
    jieba.analyse.set_idf_path(os.path.join(root_path, "data", "jieba", "idf.txt.big.txt"))
    tags = jieba.analyse.extract_tags(s, topK=20, withWeight=True)
    for x, w in tags: logger.info('%s %s' % (x, w))


def textrank():
    """
    textrank关键词提取
    :return:
    """
    s = "此外，公司拟对全资子公司吉林欧亚置业有限公司增资4.3亿元，增资后，吉林欧亚置业注册资本由7000万元增加到5亿元。吉林欧亚置业主要经营范围为房地产开发及百货零售等业务。目前在建吉林欧亚城市商业综合体项目。2013年，实现营业收入0万元，实现净利润-139.13万元。"
    tags = jieba.analyse.textrank(s, topK=20, withWeight=True)
    for x, w in tags:
        logger.info('%s %s' % (x, w))


def tokenize():
    """
    分词
    :return:
    """
    s = "周大福是创新办主任也是云计算方面的专家"
    result = jieba.tokenize(s)
    logger.info("普通模式")
    for tk in result: logger.info("word: {0} \t\t start: {1} \t\t end: {2}".format(tk[0], tk[1], tk[2]))

    logger.info("\n搜索模式")
    result = jieba.tokenize(s, mode='search')
    for tk in result: logger.info("word: {0} \t\t start: {1} \t\t end: {2}".format(tk[0], tk[1], tk[2]))


if __name__ == "__main__":
    # cut()
    # cut_for_search()
    # load_userdict()
    # dictionary()
    # add_word()
    # stop_words()
    # tf_idf()
    custorm_idf()
    # textrank()
    # tokenize()
