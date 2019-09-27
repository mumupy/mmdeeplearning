#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/24 16:16
# @Author  : ganliang
# @File    : kaggle.py
# @Desc    : kaggle预测房屋价格
import os
import time

import pandas as pd
from mxnet import nd, gluon, autograd
from mxnet.gluon import nn, loss as gloss, data as gdata

from src.config import logger, root_path

loss = gloss.L2Loss()


def get_net():
    """
    net.add(gluon.nn.Dense(128, activation=“relu”))
    net.add(gluon.nn.Dense(64, activation=“relu”))
    net.add(gluon.nn.Dropout(0.2))
    net.add(gluon.nn.Dense(1, activation=“relu”))

    batch_size = 256
    epoch = 800
    wd = 10000
    :return:
    """
    net = nn.Sequential()
    # net.add(nn.Dense(1))

    net.add(gluon.nn.Dense(128, activation="relu"))
    # net.add(gluon.nn.Dense(64, activation="relu"))
    net.add(gluon.nn.Dropout(0.2))
    net.add(gluon.nn.Dense(1))

    net.initialize()
    return net


def log_rmse(net, features, labels):
    # 将小于1的值设成1，使得取对数时数值更稳定
    clipped_preds = nd.clip(net(features), 1, float('inf'))
    rmse = nd.sqrt(2 * loss(clipped_preds.log(), labels.log()).mean())
    return rmse.asscalar()


def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    """
    数据训练
    :return:
    """
    train_ls, test_ls = [], []
    train_iter = gdata.DataLoader(gdata.ArrayDataset(train_features, train_labels), batch_size, shuffle=True)
    # 这里使用了Adam优化算法(adam,sgd)
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': learning_rate, 'wd': weight_decay})
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None: test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls


def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train, X_valid, y_valid = None, None, None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = nd.concat(X_train, X_part, dim=0)
            y_train = nd.concat(y_train, y_part, dim=0)
    return X_train, y_train, X_valid, y_valid


def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size, net=None):
    logger.info("开始 {0} 折交叉训练法".format(k))
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        if not net: net = get_net()
        data = get_k_fold_data(k, i, X_train, y_train)
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        # if i == 0:
        #     d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse', range(1, num_epochs + 1), valid_ls,['train', 'valid'])
        logger.info('fold %d, train rmse %f, valid rmse %f' % (i, train_ls[-1], valid_ls[-1]))
    return train_l_sum / k, valid_l_sum / k


def leave_one_out(X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size, net=None):
    """
    留一法(Leave-One-Out)是S折交叉验证的一种特殊情况，当S=N时交叉验证便是留一法，其中N为数据集的大小。该方法往往比较准确，但是计算量太大，比如数据集有10万个样本，那么就需要训练10万个模型
    :return:
    """
    logger.info("开始留一发模型校验")
    train_l_sum, valid_l_sum = 0, 0

    # 计算样本数量
    k = len(X_train)
    if not net: net = get_net()
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        logger.info('fold %d, train rmse %f, valid rmse %f' % (i, train_ls[-1], valid_ls[-1]))
    return train_l_sum / k, valid_l_sum / k


def hold_out():
    """
    留出发 去过拟合
    直接将数据集D划分为两个互斥的集合，其中一个集合作为训练集S，另外一个作为测试集T，即D=S∪T,S∩T=0.在S上训练出模型后，用T来评估其测试误差，作为对泛化误差的评估
    :return:
    """


def random():
    """
    给定包含N个样本的数据集TT，有放回的采样N次，得到采样集TsTs。数据集TT中样本可能在TsTs中多次，也可能不出现在TsTs。一个样本始终不在采样集中出现的概率是(1−1N)N(1−1N)N。根据：limN→∞(1−1N)N=1e=0.368limN→∞(1−1N)N=1e=0.368，因此TT中约有63.2%的样本出现在TsTs中。将TsTs用作训练集，T−TsT−Ts用作测试集。
    :return:
    """
    pass


def train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size,
                   net=None, k=0):
    if not net: net = get_net()

    # TODO 为什么模型通过k折交叉法训练完成之后 又训练一次???
    # k折交叉 每次都有1/k的数据用作测试数据，1-1/k的数据用作训练数据，所以k折训练完成之后 在全量数据训练一次，弥补k折训练遗漏的测试数据
    train_ls, _ = train(net, train_features, train_labels, None, None, num_epochs, lr, weight_decay, batch_size)

    # d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse')
    logger.info('train rmse %f' % train_ls[-1])
    preds = net(test_features).asnumpy()
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)

    current_time = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
    filename = "k{0}_lr{1}_epochs{2}_batch{3}_weight{4}_{5}".format(k, lr, num_epochs, batch_size, weight_decay,
                                                                    current_time)
    logger.info(filename)
    logger.info("\n")
    submission_dir = os.path.join(root_path, "data/kaggle_HousePrices/submission")
    if not os.path.exists(submission_dir): os.makedirs(submission_dir)

    submission_file = os.path.join(submission_dir, 'submission_{0}.csv'.format(filename))
    submission.to_csv(submission_file, index=False)


def eval_houseprice(k=10, num_epochs=100, lr=5, weight_decay=0, batch_size=64):
    """
    预测房屋价格
    :param k:  k折交叉验证法
    :param num_epochs:  训练批次大小
    :param lr:  训练率
    :param weight_decay: 权重
    :param batch_size: 每次拉去的数据量
    :return:
    """
    train_data = pd.read_csv(os.path.join(root_path, "data/kaggle_HousePrices/train.csv"))
    test_data = pd.read_csv(os.path.join(root_path, "data/kaggle_HousePrices/test.csv"))

    logger.info("\n")
    logger.info(
        "kaggle_house_pred_train k:{0} lr:{1} epochs:{2} batch:{3} weight:{4}".format(k, lr, num_epochs, batch_size,
                                                                                      weight_decay))

    logger.info("train_data length:{0}".format(len(train_data)))
    logger.info("test_data length:{0}".format(len(test_data)))

    top_features = train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]]
    logger.info(top_features)

    all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

    numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
    all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
    # 标准化后，每个特征的均值变为0，所以可以直接用0来替换缺失值
    all_features[numeric_features] = all_features[numeric_features].fillna(0)

    all_features = pd.get_dummies(all_features, dummy_na=True)

    n_train = train_data.shape[0]
    train_features = nd.array(all_features[:n_train].values)
    test_features = nd.array(all_features[n_train:].values)
    train_labels = nd.array(train_data.SalePrice.values).reshape((-1, 1))

    # net = get_net()

    # train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
    train_l, valid_l = leave_one_out(train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
    logger.info('%d-fold validation: avg train rmse %f, avg valid rmse %f' % (k, train_l, valid_l))

    train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size,
                   k=k)


if __name__ == "__main__":
    eval_houseprice()
    # for i in range(1):
    #     eval_houseprice(k=10, batch_size=64, lr=5)
