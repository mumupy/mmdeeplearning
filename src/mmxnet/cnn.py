#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/27 9:05
# @Author  : ganliang
# @File    : convolutionallayer.py
# @Desc    : cnn卷积神经网络 卷积层 卷积神经网络（convolutional neural network）是含有卷积层（convolutional layer）的神经网络
import os
import sys

import d2lzh as d2l
import mxnet
from mxnet import autograd, nd, gluon, init
from mxnet.gluon import nn, data as gdata

from src.config import logger


class Conv2D(nn.Block):
    def __init__(self, kernel_size, **kwargs):
        super(Conv2D, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=kernel_size)
        self.bias = self.params.get('bias', shape=(1,))

    def forward(self, x):
        return corr2d(x, self.weight.data()) + self.bias.data()


def corr2d(X, K):  # 本函数已保存在d2lzh包中方便以后使用
    h, w = K.shape
    Y = nd.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y


def cal_corr2d():
    X = nd.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    K = nd.array([[0, 1], [2, 3]])
    Y = corr2d(X, K)
    logger.info("卷积核:%s" % Y)


def convolutional():
    X = nd.ones((6, 8))
    X[:, 2:6] = 0

    conv2d = nn.Conv2D(1, kernel_size=(1, 2))
    conv2d.initialize()

    # 二维卷积层使用4维输入输出，格式为(样本, 通道, 高, 宽)，这里批量大小（批量中的样本数）和通
    # 道数均为1
    X = nd.ones((6, 8))
    X[:, 2:6] = 0
    K = nd.array([[1, -1]])
    Y = corr2d(X, K)
    logger.info("卷积核: %s" % Y)

    X = X.reshape((1, 1, 6, 8))
    Y = Y.reshape((1, 1, 6, 7))

    for i in range(10):
        with autograd.record():
            Y_hat = conv2d(X)
            l = (Y_hat - Y) ** 2
        l.backward()
        # 简单起见，这里忽略了偏差
        conv2d.weight.data()[:] -= 3e-2 * conv2d.weight.grad()
        if (i + 1) % 2 == 0:
            logger.info('batch %d, loss %.3f' % (i + 1, l.sum().asscalar()))

    logger.info(conv2d.weight.data().reshape((1, 2)))


def comp_conv2d(conv2d, X):
    conv2d.initialize()
    # (1, 1)代表批量大小和通道数（“多输入通道和多输出通道”一节将介绍）均为1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:])  # 排除不关心的前两维：批量和通道


def lenet():
    """

    :return:
    """

    net = nn.Sequential()
    net.add(nn.Conv2D(channels=6, kernel_size=5, activation='sigmoid'),
            nn.MaxPool2D(pool_size=2, strides=2),
            nn.Conv2D(channels=16, kernel_size=5, activation='sigmoid'),
            nn.MaxPool2D(pool_size=2, strides=2),
            # Dense会默认将(批量大小, 通道, 高, 宽)形状的输入转换成
            # (批量大小, 通道 * 高 * 宽)形状的输入
            nn.Dense(120, activation='sigmoid'),
            nn.Dense(84, activation='sigmoid'),
            nn.Dense(10))
    X = nd.random.uniform(shape=(1, 1, 28, 28))
    net.initialize()
    for layer in net:
        X = layer(X)
        print(layer.name, 'output shape:\t', X.shape)

    lr, num_epochs, batch_size = 0.9, 5, 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

    net.initialize(force_reinit=True, init=init.Xavier())
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})

    d2l.train_ch5(net, train_iter, test_iter, batch_size, trainer, mxnet.cpu(), num_epochs)


def alexnet():
    """
    alexnet深度学习
    :return:
    """

    def load_data_fashion_mnist(batch_size, resize=None, root=os.path.join(
        '~', '.mxnet', 'datasets', 'fashion-mnist')):
        root = os.path.expanduser(root)  # 展开用户路径'~'
        transformer = []
        if resize:
            transformer += [gdata.vision.transforms.Resize(resize)]
        transformer += [gdata.vision.transforms.ToTensor()]
        transformer = gdata.vision.transforms.Compose(transformer)
        mnist_train = gdata.vision.FashionMNIST(root=root, train=True)
        mnist_test = gdata.vision.FashionMNIST(root=root, train=False)
        num_workers = 0 if sys.platform.startswith('win32') else 4
        train_iter = gdata.DataLoader(
            mnist_train.transform_first(transformer), batch_size, shuffle=True,
            num_workers=num_workers)
        test_iter = gdata.DataLoader(
            mnist_test.transform_first(transformer), batch_size, shuffle=False,
            num_workers=num_workers)
        return train_iter, test_iter

    net = nn.Sequential()
    # 使用较大的11 x 11窗口来捕获物体。同时使用步幅4来较大幅度减小输出高和宽。这里使用的输出通
    # 道数比LeNet中的也要大很多
    net.add(nn.Conv2D(96, kernel_size=11, strides=4, activation='relu'),
            nn.MaxPool2D(pool_size=3, strides=2),
            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
            nn.Conv2D(256, kernel_size=5, padding=2, activation='relu'),
            nn.MaxPool2D(pool_size=3, strides=2),
            # 连续3个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数。
            # 前两个卷积层后不使用池化层来减小输入的高和宽
            nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
            nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
            nn.Conv2D(256, kernel_size=3, padding=1, activation='relu'),
            nn.MaxPool2D(pool_size=3, strides=2),
            # 这里全连接层的输出个数比LeNet中的大数倍。使用丢弃层来缓解过拟合
            nn.Dense(4096, activation="relu"), nn.Dropout(0.5),
            nn.Dense(4096, activation="relu"), nn.Dropout(0.5),
            # 输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
            nn.Dense(10))

    X = nd.random.uniform(shape=(1, 1, 224, 224))
    net.initialize()
    for layer in net:
        X = layer(X)
        print(layer.name, 'output shape:\t', X.shape)

    batch_size = 128
    # 如出现“out of memory”的报错信息，可减小batch_size或resize
    train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)

    lr, num_epochs, ctx = 0.01, 5, d2l.try_gpu()
    net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
    d2l.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)


def vgg():
    """

    :return:
    """

    def vgg_block(num_convs, num_channels):
        blk = nn.Sequential()
        for _ in range(num_convs):
            blk.add(nn.Conv2D(num_channels, kernel_size=3,
                              padding=1, activation='relu'))
        blk.add(nn.MaxPool2D(pool_size=2, strides=2))
        return blk

    def vgg(conv_arch):
        net = nn.Sequential()
        # 卷积层部分
        for (num_convs, num_channels) in conv_arch:
            net.add(vgg_block(num_convs, num_channels))
        # 全连接层部分
        net.add(nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
                nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
                nn.Dense(10))
        return net

    conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
    net = vgg(conv_arch)

    net.initialize()
    X = nd.random.uniform(shape=(1, 1, 224, 224))
    for blk in net:
        X = blk(X)
        print(blk.name, 'output shape:\t', X.shape)

    ratio = 4
    small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
    net = vgg(small_conv_arch)

    lr, num_epochs, batch_size, ctx = 0.05, 5, 128, d2l.try_gpu()
    net.initialize(ctx=ctx, init=init.Xavier())
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
    d2l.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx,
                  num_epochs)


def nin():
    """

    :return:
    """

    def nin_block(num_channels, kernel_size, strides, padding):
        blk = nn.Sequential()
        blk.add(nn.Conv2D(num_channels, kernel_size, strides, padding, activation='relu'),
                nn.Conv2D(num_channels, kernel_size=1, activation='relu'),
                nn.Conv2D(num_channels, kernel_size=1, activation='relu'))
        return blk

    net = nn.Sequential()
    net.add(nin_block(96, kernel_size=11, strides=4, padding=0),
            nn.MaxPool2D(pool_size=3, strides=2),
            nin_block(256, kernel_size=5, strides=1, padding=2),
            nn.MaxPool2D(pool_size=3, strides=2),
            nin_block(384, kernel_size=3, strides=1, padding=1),
            nn.MaxPool2D(pool_size=3, strides=2), nn.Dropout(0.5),
            # 标签类别数是10
            nin_block(10, kernel_size=3, strides=1, padding=1),
            # 全局平均池化层将窗口形状自动设置成输入的高和宽
            nn.GlobalAvgPool2D(),
            # 将四维的输出转成二维的输出，其形状为(批量大小, 10)
            nn.Flatten())
    X = nd.random.uniform(shape=(1, 1, 224, 224))
    net.initialize()
    for layer in net:
        X = layer(X)
        print(layer.name, 'output shape:\t', X.shape)

    lr, num_epochs, batch_size, ctx = 0.1, 5, 128, d2l.try_gpu()
    net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
    d2l.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)


def googlenet():
    """
    GoogLeNet吸收了NiN中网络串联网络的思想，并在此基础上做了很大改进。在随后的几年里，研究人员对GoogLeNet进行了数次改进，本节将介绍这个模型系列的第一个版本。
    :return:
    """

    class Inception(nn.Block):
        # c1 - c4为每条线路里的层的输出通道数
        def __init__(self, c1, c2, c3, c4, **kwargs):
            super(Inception, self).__init__(**kwargs)
            # 线路1，单1 x 1卷积层
            self.p1_1 = nn.Conv2D(c1, kernel_size=1, activation='relu')
            # 线路2，1 x 1卷积层后接3 x 3卷积层
            self.p2_1 = nn.Conv2D(c2[0], kernel_size=1, activation='relu')
            self.p2_2 = nn.Conv2D(c2[1], kernel_size=3, padding=1,
                                  activation='relu')
            # 线路3，1 x 1卷积层后接5 x 5卷积层
            self.p3_1 = nn.Conv2D(c3[0], kernel_size=1, activation='relu')
            self.p3_2 = nn.Conv2D(c3[1], kernel_size=5, padding=2,
                                  activation='relu')
            # 线路4，3 x 3最大池化层后接1 x 1卷积层
            self.p4_1 = nn.MaxPool2D(pool_size=3, strides=1, padding=1)
            self.p4_2 = nn.Conv2D(c4, kernel_size=1, activation='relu')

        def forward(self, x):
            p1 = self.p1_1(x)
            p2 = self.p2_2(self.p2_1(x))
            p3 = self.p3_2(self.p3_1(x))
            p4 = self.p4_2(self.p4_1(x))
            return nd.concat(p1, p2, p3, p4, dim=1)  # 在通道维上连结输出

    b1 = nn.Sequential()
    b1.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3, activation='relu'),
           nn.MaxPool2D(pool_size=3, strides=2, padding=1))

    b2 = nn.Sequential()
    b2.add(nn.Conv2D(64, kernel_size=1, activation='relu'),
           nn.Conv2D(192, kernel_size=3, padding=1, activation='relu'),
           nn.MaxPool2D(pool_size=3, strides=2, padding=1))

    b3 = nn.Sequential()
    b3.add(Inception(64, (96, 128), (16, 32), 32),
           Inception(128, (128, 192), (32, 96), 64),
           nn.MaxPool2D(pool_size=3, strides=2, padding=1))

    b4 = nn.Sequential()
    b4.add(Inception(192, (96, 208), (16, 48), 64),
           Inception(160, (112, 224), (24, 64), 64),
           Inception(128, (128, 256), (24, 64), 64),
           Inception(112, (144, 288), (32, 64), 64),
           Inception(256, (160, 320), (32, 128), 128),
           nn.MaxPool2D(pool_size=3, strides=2, padding=1))

    b5 = nn.Sequential()
    b5.add(Inception(256, (160, 320), (32, 128), 128),
           Inception(384, (192, 384), (48, 128), 128),
           nn.GlobalAvgPool2D())

    net = nn.Sequential()
    net.add(b1, b2, b3, b4, b5, nn.Dense(10))

    X = nd.random.uniform(shape=(1, 1, 96, 96))
    net.initialize()
    for layer in net:
        X = layer(X)
        print(layer.name, 'output shape:\t', X.shape)

    lr, num_epochs, batch_size, ctx = 0.1, 5, 128, d2l.try_gpu()
    net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
    d2l.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)


def batchnormalization():
    """
    批量归一化利用小批量上的均值和标准差，不断调整神经网络的中间输出，从而使整个神经网络在各层的中间输出的数值更稳定
    :return:
    """

    def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
        # 通过autograd来判断当前模式是训练模式还是预测模式
        if not autograd.is_training():
            # 如果是在预测模式下，直接使用传入的移动平均所得的均值和方差
            X_hat = (X - moving_mean) / nd.sqrt(moving_var + eps)
        else:
            assert len(X.shape) in (2, 4)
            if len(X.shape) == 2:
                # 使用全连接层的情况，计算特征维上的均值和方差
                mean = X.mean(axis=0)
                var = ((X - mean) ** 2).mean(axis=0)
            else:
                # 使用二维卷积层的情况，计算通道维上（axis=1）的均值和方差。这里我们需要保持
                # X的形状以便后面可以做广播运算
                mean = X.mean(axis=(0, 2, 3), keepdims=True)
                var = ((X - mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)
            # 训练模式下用当前的均值和方差做标准化
            X_hat = (X - mean) / nd.sqrt(var + eps)
            # 更新移动平均的均值和方差
            moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
            moving_var = momentum * moving_var + (1.0 - momentum) * var
        Y = gamma * X_hat + beta  # 拉伸和偏移
        return Y, moving_mean, moving_var

    class BatchNorm(nn.Block):
        def __init__(self, num_features, num_dims, **kwargs):
            super(BatchNorm, self).__init__(**kwargs)
            if num_dims == 2:
                shape = (1, num_features)
            else:
                shape = (1, num_features, 1, 1)
            # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成1和0
            self.gamma = self.params.get('gamma', shape=shape, init=init.One())
            self.beta = self.params.get('beta', shape=shape, init=init.Zero())
            # 不参与求梯度和迭代的变量，全在内存上初始化成0
            self.moving_mean = nd.zeros(shape)
            self.moving_var = nd.zeros(shape)

        def forward(self, X):
            # 如果X不在内存上，将moving_mean和moving_var复制到X所在显存上
            if self.moving_mean.context != X.context:
                self.moving_mean = self.moving_mean.copyto(X.context)
                self.moving_var = self.moving_var.copyto(X.context)
            # 保存更新过的moving_mean和moving_var
            Y, self.moving_mean, self.moving_var = batch_norm(
                X, self.gamma.data(), self.beta.data(), self.moving_mean,
                self.moving_var, eps=1e-5, momentum=0.9)
            return Y

    net = nn.Sequential()
    net.add(nn.Conv2D(6, kernel_size=5),
            BatchNorm(6, num_dims=4),
            nn.Activation('sigmoid'),
            nn.MaxPool2D(pool_size=2, strides=2),
            nn.Conv2D(16, kernel_size=5),
            BatchNorm(16, num_dims=4),
            nn.Activation('sigmoid'),
            nn.MaxPool2D(pool_size=2, strides=2),
            nn.Dense(120),
            BatchNorm(120, num_dims=2),
            nn.Activation('sigmoid'),
            nn.Dense(84),
            BatchNorm(84, num_dims=2),
            nn.Activation('sigmoid'),
            nn.Dense(10))

    lr, num_epochs, batch_size, ctx = 1.0, 5, 256, d2l.try_gpu()
    net.initialize(ctx=ctx, init=init.Xavier())
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    d2l.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx,
                  num_epochs)

    net[1].gamma.data().reshape((-1,)), net[1].beta.data().reshape((-1,))


def batchnormalization_simple():
    """
    Gluon中nn模块定义的BatchNorm类使用起来更加简单。它不需要指定自己定义的BatchNorm类中所需的num_features和num_dims参数值。
    在Gluon中，这些参数值都将通过延后初始化而自动获取。下面我们用Gluon实现使用批量归一化的LeNet。
    :return:
    """

    net = nn.Sequential()
    net.add(nn.Conv2D(6, kernel_size=5),
            nn.BatchNorm(),
            nn.Activation('sigmoid'),
            nn.MaxPool2D(pool_size=2, strides=2),
            nn.Conv2D(16, kernel_size=5),
            nn.BatchNorm(),
            nn.Activation('sigmoid'),
            nn.MaxPool2D(pool_size=2, strides=2),
            nn.Dense(120),
            nn.BatchNorm(),
            nn.Activation('sigmoid'),
            nn.Dense(84),
            nn.BatchNorm(),
            nn.Activation('sigmoid'),
            nn.Dense(10))

    lr, num_epochs, batch_size, ctx = 1.0, 5, 256, d2l.try_gpu()
    net.initialize(ctx=ctx, init=init.Xavier())
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    d2l.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)


def resnet():
    """
    残差块通过跨层的数据通道从而能够训练出有效的深度神经网络。
    :return:
    """

    class Residual(nn.Block):  # 本类已保存在d2lzh包中方便以后使用
        def __init__(self, num_channels, use_1x1conv=False, strides=1, **kwargs):
            super(Residual, self).__init__(**kwargs)
            self.conv1 = nn.Conv2D(num_channels, kernel_size=3, padding=1,
                                   strides=strides)
            self.conv2 = nn.Conv2D(num_channels, kernel_size=3, padding=1)
            if use_1x1conv:
                self.conv3 = nn.Conv2D(num_channels, kernel_size=1,
                                       strides=strides)
            else:
                self.conv3 = None
            self.bn1 = nn.BatchNorm()
            self.bn2 = nn.BatchNorm()

        def forward(self, X):
            Y = nd.relu(self.bn1(self.conv1(X)))
            Y = self.bn2(self.conv2(Y))
            if self.conv3:
                X = self.conv3(X)
            return nd.relu(Y + X)

    def resnet_block(num_channels, num_residuals, first_block=False):
        blk = nn.Sequential()
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.add(Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                blk.add(Residual(num_channels))
        return blk

    net = nn.Sequential()
    net.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3),
            nn.BatchNorm(), nn.Activation('relu'),
            nn.MaxPool2D(pool_size=3, strides=2, padding=1))

    net.add(resnet_block(64, 2, first_block=True),
            resnet_block(128, 2),
            resnet_block(256, 2),
            resnet_block(512, 2))

    net.add(nn.GlobalAvgPool2D(), nn.Dense(10))

    X = nd.random.uniform(shape=(1, 1, 224, 224))
    net.initialize()
    for layer in net:
        X = layer(X)
        print(layer.name, 'output shape:\t', X.shape)

    lr, num_epochs, batch_size, ctx = 0.05, 5, 256, d2l.try_gpu()
    net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
    d2l.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)


def densenet():
    """
    DenseNet的主要构建模块是稠密块（dense block）和过渡层（transition layer）。前者定义了输入和输出是如何连结的，后者则用来控制
    通道数，使之不过大。
    :return:
    """

    class DenseBlock(nn.Block):
        def __init__(self, num_convs, num_channels, **kwargs):
            super(DenseBlock, self).__init__(**kwargs)
            self.net = nn.Sequential()
            for _ in range(num_convs):
                self.net.add(conv_block(num_channels))

        def forward(self, X):
            for blk in self.net:
                Y = blk(X)
                X = nd.concat(X, Y, dim=1)  # 在通道维上将输入和输出连结
            return X

    def conv_block(num_channels):
        blk = nn.Sequential()
        blk.add(nn.BatchNorm(), nn.Activation('relu'),
                nn.Conv2D(num_channels, kernel_size=3, padding=1))
        return blk

    def transition_block(num_channels):
        blk = nn.Sequential()
        blk.add(nn.BatchNorm(), nn.Activation('relu'),
                nn.Conv2D(num_channels, kernel_size=1),
                nn.AvgPool2D(pool_size=2, strides=2))
        return blk

    net = nn.Sequential()
    net.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3),
            nn.BatchNorm(), nn.Activation('relu'),
            nn.MaxPool2D(pool_size=3, strides=2, padding=1))

    num_channels, growth_rate = 64, 32  # num_channels为当前的通道数
    num_convs_in_dense_blocks = [4, 4, 4, 4]

    for i, num_convs in enumerate(num_convs_in_dense_blocks):
        net.add(DenseBlock(num_convs, growth_rate))
        # 上一个稠密块的输出通道数
        num_channels += num_convs * growth_rate
        # 在稠密块之间加入通道数减半的过渡层
        if i != len(num_convs_in_dense_blocks) - 1:
            num_channels //= 2
            net.add(transition_block(num_channels))

    net.add(nn.BatchNorm(), nn.Activation('relu'), nn.GlobalAvgPool2D(), nn.Dense(10))

    lr, num_epochs, batch_size, ctx = 0.1, 5, 256, d2l.try_gpu()
    net.initialize(ctx=ctx, init=init.Xavier())
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
    d2l.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)


if __name__ == "__main__":
    # cal_corr2d()
    # convolutional()
    # lenet()
    # alexnet()
    # vgg()
    # nin()
    # googlenet()
    # resnet()
    densenet()
