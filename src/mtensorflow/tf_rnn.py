#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/26 9:40
# @Author  : ganliang
# @File    : tf_rnn.py
# @Desc    : rnn循环神经网络
import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

from src.config import root_path


def mnist_runn():
    """
    nmt翻译
    :return:
    """
    lr = 0.0001
    training_iters = 100000
    batch_size = 128
    epoch = 10

    # 神经网络的参数 n_input = 28 # 输入层的 n n_steps = 28 # 28 长度
    n_inputs = 28  # 输入层的 n n_steps = 28 # 28 长度
    n_hidden_units=128
    n_steps = 28
    n_classes = 10  # 输出的数量，即分类的类别，0～9 个数字，共有 10 个

    # 输入数据占位符
    x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    y = tf.placeholder(tf.float32, [None, n_classes])
    # 定义权重
    weights = {
        'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
        'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
    }
    biases = {
        'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
        'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
    }

    def RNN(X, weights, biases):
        X = tf.reshape(X, [-1, n_inputs])
        # 进入隐藏层
        # X_in = (128 batch * 28 steps, 128 hidden)
        X_in = tf.matmul(X, weights['in']) + biases['in']
        # X_in ==> (128 batch, 28 steps, 128 hidden)
        X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
        # 初始化为零值，lstm 单元由两个部分组成：(c_state, h_state)
        init_state = lstm_cell.zero_state(n_hidden_units, dtype=tf.float32)
        # dynamic_rnn 接收张量(batch, steps, inputs)或者(steps, batch, inputs)作为 X_in
        outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)
        results = tf.matmul(final_state[1], weights['out']) + biases['out']
        return results

    pred = RNN(x, weights, biases)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1)), tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        dataset = input_data.read_data_sets(os.path.join(root_path, "data", "fashionMNIST"), one_hot=True)
        batch_count = training_iters // batch_size
        for epoch_index in range(epoch):
            for batch_index in range(batch_count):
                batch_xs, batch_ys = dataset.train.next_batch(batch_size)
                batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
                _, l, ac = sess.run([optimizer, loss, accuracy],
                                    feed_dict={x: batch_xs,
                                               y: batch_ys})
                print("Train: Epoch={0},Batch={1},loss={2},accuracy={3}".format(epoch_index, batch_index, l, ac))
            # 验证数据
            l, ac = sess.run([loss, accuracy],
                             feed_dict={x: dataset.validation.images.reshape([-1, n_steps, n_inputs])[0:batch_size],
                                        y: dataset.validation.labels[0:batch_size]})
            print("Validation: Epoch={0},loss={1},accuracy={2}".format(epoch_index, l, ac))
        # 测试数据
        l, ac = sess.run([loss, accuracy],
                         feed_dict={x: dataset.test.images.reshape([-1, n_steps, n_inputs])[0:batch_size],
                                    y: dataset.test.labels[0:batch_size]})
        print("Test: loss={0},accuracy={1}".format(l, ac))


def mnist_encode():
    # 设置训练超格数
    learning_rate = 0.01  # 学习率
    training_epochs = 20  # 训练的轮数
    batch_size = 256  # 每次训练的数据多少
    display_step = 1  # 每隔多少轮显示一次训练结果

    examples_to_show = 10

    n_hidden_1 = 256
    n_hidden_2 = 128
    n_input = 784

    weights = {'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
               'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
               'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
               'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])), }
    biases = {'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
              'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
              'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
              'decoder_b2': tf.Variable(tf.random_normal([n_input])), }

    def encoder(x):
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
        return layer_2

    def decoder(x):
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
        return layer_2

    X = tf.placeholder("float", [None, n_input])

    decoder_op = decoder(encoder(X))

    cost = tf.reduce_mean(tf.pow(X - decoder_op, 2))
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    dataset = input_data.read_data_sets(os.path.join(root_path, "data", "fashionMNIST"))
    with tf.Session() as sess:
        sess.run(init)
        total_batch = int(dataset.train.num_examples / batch_size)  # 开始训练 for epoch in range(training_epochs):
        for epoch in range(training_epochs):
            for i in range(total_batch):
                batch_xs, batch_ys = dataset.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
                if epoch % display_step == 0:
                    print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c))
        print("Optimization Finished!")

        encode_decode = sess.run(decoder_op, feed_dict={X: dataset.test.images[:examples_to_show]})
        f, a = plt.subplots(2, 10, figsize=(10, 2))
        for i in range(examples_to_show):
            a[0][i].imshow(np.reshape(dataset.test.images[i], (28, 28)))
            a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
        f.show()
        plt.draw()
        plt.waitforbuttonpress()


if __name__ == "__main__":
    mnist_runn()
    # mnist_encode()
