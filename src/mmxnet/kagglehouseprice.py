#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/24 16:16
# @Author  : ganliang
# @File    : kaggle.py
# @Desc    : kaggle预测房屋价格
import d2lzh as d2l
import pandas as pd
from mxnet import nd, gluon, autograd
from mxnet.gluon import nn, loss as gloss, data as gdata

from src.config import logger


def eval_houseprice():
    """
    预测房屋价格
    :return:
    """

    def get_net():
        net = nn.Sequential()
        net.add(nn.Dense(1))
        net.initialize()
        return net

    def log_rmse(net, features, labels):
        # 将小于1的值设成1，使得取对数时数值更稳定
        clipped_preds = nd.clip(net(features), 1, float('inf'))
        rmse = nd.sqrt(2 * loss(clipped_preds.log(), labels.log()).mean())
        return rmse.asscalar()

    def train(net, train_features, train_labels, test_features, test_labels,
              num_epochs, learning_rate, weight_decay, batch_size):
        train_ls, test_ls = [], []
        train_iter = gdata.DataLoader(gdata.ArrayDataset(train_features, train_labels), batch_size, shuffle=True)
        # 这里使用了Adam优化算法
        trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': learning_rate, 'wd': weight_decay})
        for epoch in range(num_epochs):
            for X, y in train_iter:
                with autograd.record():
                    l = loss(net(X), y)
                l.backward()
                trainer.step(batch_size)
            train_ls.append(log_rmse(net, train_features, train_labels))
            if test_labels is not None:
                test_ls.append(log_rmse(net, test_features, test_labels))
        return train_ls, test_ls

    def get_k_fold_data(k, i, X, y):
        assert k > 1
        fold_size = X.shape[0] // k
        X_train, y_train = None, None
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

    def k_fold(k, X_train, y_train, num_epochs,
               learning_rate, weight_decay, batch_size):
        train_l_sum, valid_l_sum = 0, 0
        for i in range(k):
            data = get_k_fold_data(k, i, X_train, y_train)
            net = get_net()
            train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
            train_l_sum += train_ls[-1]
            valid_l_sum += valid_ls[-1]
            if i == 0:
                d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse', range(1, num_epochs + 1), valid_ls,
                             ['train', 'valid'])
            print('fold %d, train rmse %f, valid rmse %f' % (i, train_ls[-1], valid_ls[-1]))
        return train_l_sum / k, valid_l_sum / k

    train_data = pd.read_csv('../data/kaggle_house_pred_train.csv')
    test_data = pd.read_csv('../data/kaggle_house_pred_test.csv')

    logger.info("kaggle_house_pred_train")
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

    loss = gloss.L2Loss()

    k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
    train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
    print('%d-fold validation: avg train rmse %f, avg valid rmse %f'
          % (k, train_l, valid_l))


if __name__ == "__main__":
    eval_houseprice()
