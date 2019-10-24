反向传播(BPN)
=============

反向传播（BPN）算法是神经网络中研究最多、使用最多的算法之一，它用于将输出层中的误差传播到隐藏层的神经元，然后用于更新权重。

学习 BPN 算法可以分成以下两个过程：
- 正向传播：输入被馈送到网络，信号从输入层通过隐藏层传播到输出层。在输出层，计算误差和损失函数。
- 反向传播：在反向传播中，首先计算输出层神经元损失函数的梯度，然后计算隐藏层神经元损失函数的梯度。接下来用梯度更新权重。


数学推理
::::::::

首先给网络提供 M 个训练对（X，Y），X 为输入，Y 为期望的输出。输入通过激活函数 g(h) 和隐藏层传播到输出层。输出 Yhat 是网络的输出，得到 error=Y-Yhat。其损失函数 J(W) 如下：

.. image:: /_static/imgs/overview/bpn1.png

其中，i 取遍所有输出层的神经元（1 到 N）。然后可以使用 J(W) 的梯度并使用链式法则求导，来计算连接第 i 个输出层神经元到第 j 个隐藏层神经元的权重 Wij 的变化：

.. image:: /_static/imgs/overview/bpn2.png

这里，Oj 是隐藏层神经元的输出，h 表示隐藏层的输入值。这很容易理解，但现在怎么更新连接第 n 个隐藏层的神经元 k 到第 n+1 个隐藏层的神经元 j 的权值 Wjk？过程是相同的：将使用损失函数的梯度和链式法则求导，但这次计算 Wjk：

.. image:: /_static/imgs/overview/bpn3.png

代码实现
::::::::

::

    def _sigmaprime(X):
        return tf.multiply(tf.sigmoid(X), tf.subtract(tf.constant(1.0), tf.sigmoid(X)))


    def _multilayer_perceptron(X, weights, biases):
        h_layer_1 = tf.add(tf.matmul(X, weights["h1"]), biases["h1"])
        out_layer_1 = tf.sigmoid(h_layer_1)

        h_out = tf.matmul(out_layer_1, weights["out"]) + biases["out"]
        return tf.sigmoid(h_out), h_out, out_layer_1, h_layer_1


    def bpn(epochs=10, batch_size=1000, learning_rate=0.01, hidden=30):
        """
        反向传播实现
        :return:
        """
        mnist_path = os.path.join(root_path, "data", "fashionMNIST")
        mnist_data = input_data.read_data_sets(mnist_path, one_hot=True)
        train_data = mnist_data.train
        test_data = mnist_data.test

        sample_count = train_data.num_examples
        feature_count = train_data.images.shape[1]

        label_count = train_data.labels.shape[1]

        X = tf.placeholder(tf.float32, shape=(None, feature_count))
        Y = tf.placeholder(tf.float32, shape=(None, label_count))

        weights = {
            "h1": tf.Variable(tf.random_normal(shape=(feature_count, hidden), seed=0)),
            "out": tf.Variable(tf.random_normal(shape=(hidden, label_count), seed=0))
        }
        baises = {
            "h1": tf.Variable(tf.random_normal(shape=(1, hidden), seed=0)),
            "out": tf.Variable(tf.random_normal(shape=(1, label_count), seed=0))
        }
        Y_that, h_out, out_layer_1, h_layer_1 = _multilayer_perceptron(X, weights, baises)

        # 反向传播
        error = Y_that - Y
        delta_2 = tf.multiply(error, _sigmaprime(h_out))
        delta_w_2 = tf.matmul(tf.transpose(out_layer_1), delta_2)

        wtd_error = tf.matmul(delta_2, tf.transpose(weights["out"]))
        delta_1 = tf.multiply(wtd_error, _sigmaprime(h_layer_1))
        delta_w_1 = tf.matmul(tf.transpose(X), delta_1)

        eta = tf.constant(learning_rate)

        step = [tf.assign(weights["h1"], tf.subtract(weights["h1"], tf.multiply(eta, delta_w_1))),
                tf.assign(baises["h1"], tf.subtract(baises["h1"], tf.multiply(eta, tf.reduce_mean(delta_1, axis=[0])))),
                tf.assign(weights["out"], tf.subtract(weights["out"], tf.multiply(eta, delta_w_2))),
                tf.assign(baises["out"], tf.subtract(baises["out"], tf.multiply(eta, tf.reduce_mean(delta_2, axis=[0]))))]

        acct_mat = tf.equal(tf.argmax(Y_that, 1), tf.argmax(Y, 1))
        with tf.name_scope("accuracy"):
            accuracy = tf.reduce_mean(tf.cast(acct_mat, tf.float32))
            tf.summary.scalar("accuracy", accuracy)

        init = tf.global_variables_initializer()
        summary_ops = tf.summary.merge_all()
        acc_trains, acc_tests = [], []
        with tf.Session() as sess:
            sess.run(init)
            writer = tf.summary.FileWriter("bpn", graph=sess.graph)
            for i in range(epochs):
                batch_count = sample_count // batch_size
                for j in range(batch_count):
                    batch_trains, batch_lables = mnist_data.train.next_batch(batch_size)
                    _, summary_str = sess.run([step, summary_ops], feed_dict={X: batch_trains, Y: batch_lables})
                    writer.add_summary(summary_str, i * batch_size + j)
                # 训练数据评估值
                acc_train = sess.run(accuracy, feed_dict={X: train_data.images, Y: train_data.labels})
                # 测试数据评估值
                acc_test = sess.run(accuracy, feed_dict={X: test_data.images, Y: test_data.labels})
                logger.info("epoll {0} train accuracy {1} test accuracy {2}".format(i, acc_train, acc_test))
                acc_trains.append(acc_train)
                acc_tests.append(acc_test)
            writer.close()

        plt.plot(list(range(epochs)), acc_trains, "bo", label="train accuracy")
        plt.plot(list(range(epochs)), acc_tests, "r", label="test accuracy")
        plt.xlabel("epoch")
        plt.xlabel("accuracy")
        plt.title("accuracy train/test")
        plt.legend()
        plt.show()

输出结果图:

.. image:: /_static/imgs/overview/bpn_plt.png