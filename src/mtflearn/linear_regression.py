#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/12/3 21:00
# @Author  : ganliang
# @File    : linear_regression.py
# @Desc    : 线性回归

import tflearn


def linear_regression():
    # 回归数据
    X = [3.38, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167, 7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1]
    Y = [1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221, 2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3]

    # 线性回归图
    input_ = tflearn.input_data(shape=[None])
    linear = tflearn.single_unit(input_)
    # 优化器，目标（`optimizer`）和评价指标（`metric`）
    regression = tflearn.regression(linear, optimizer='sgd', loss='mean_square', metric='R2', learning_rate=0.01)
    # 使用`DNN`（深度神经网络）模型类训练模型
    m = tflearn.DNN(regression)
    m.fit(X, Y, n_epoch=1000, show_metric=True, snapshot_epoch=False)

    print("回归结果：")
    # `get_weights`方法获取模型的权重值
    print("Y = " + str(m.get_weights(linear.W)) + "*X + " + str(m.get_weights(linear.b)))

    print("x的测试预测 = 3.2, 3.3, 3.4:")
    print(m.predict([3.2, 3.3, 3.4]))

linear_regression()