#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/21 16:38
# @Author  : ganliang
# @File    : matplotlib.py
# @Desc    : matplotlib图形绘制

import matplotlib.font_manager
import numpy as np
from matplotlib import pyplot as plt


def show_plot():
    # 绘制2x+5公式图
    x = np.arange(1, 11)
    y = 2 * x + 5
    plt.title("Matplotlib demo")
    plt.xlabel("x axis caption")
    plt.ylabel("y axis caption")
    plt.plot(x, y)
    plt.show()


def show_system_font():
    # 字体展示
    a = sorted([f.name for f in matplotlib.font_manager.fontManager.ttflist])
    for i in a:
        print(i)


def show_font_plot():
    """
    绘制中文标题
    :return:
    """
    zhfont1 = matplotlib.font_manager.FontProperties(fname="SimHei.ttf")
    # zhfont1 = matplotlib.font_manager.FontProperties(family="STFangsong")
    x = np.arange(1, 11)
    y = 2 * x + 5
    plt.title(u"菜鸟教程 - 测试", fontproperties=zhfont1)

    # fontproperties 设置中文显示，fontsize 设置字体大小
    plt.xlabel(u"x 轴", fontproperties=zhfont1)
    plt.ylabel(u"y 轴", fontproperties=zhfont1)
    plt.plot(x, y)
    plt.show()


def show_dot_plot():
    """
    绘制点图
    :return:
    """
    x = np.arange(1, 11)
    y = 2 * x + 5
    plt.title("Matplotlib demo")
    plt.xlabel("x axis caption")
    plt.ylabel("y axis caption")
    plt.plot(x, y, "ob")
    plt.show()


def show_sin_plot():
    """
    绘制正弦值
    :return:
    """
    # 计算正弦曲线上点的 x 和 y 坐标
    x = np.arange(0, 3 * np.pi, 0.1)
    y = np.sin(x)
    plt.title("sine wave form")
    # 使用 matplotlib 来绘制点
    plt.plot(x, y, "ob")
    plt.show()


def show_sincos_plot():
    x = np.arange(0, 3 * np.pi, 0.1)
    y_sin = np.sin(x)
    y_cos = np.cos(x)
    # 建立 subplot 网格，高为 2，宽为 1
    # 激活第一个 subplot
    plt.subplot(2, 1, 1)
    # 绘制第一个图像
    plt.plot(x, y_sin)
    plt.title('Sine')
    # 将第二个 subplot 激活，并绘制第二个图像
    plt.subplot(2, 1, 2)
    plt.plot(x, y_cos)
    plt.title('Cosine')
    # 展示图像
    plt.show()


def show_bar_plot():
    x = [5, 8, 10]
    y = [12, 16, 6]
    x2 = [6, 9, 11]
    y2 = [6, 15, 7]
    plt.bar(x, y, align='center')
    plt.bar(x2, y2, color='g', align='center')
    plt.title('Bar graph')
    plt.ylabel('Y axis')
    plt.xlabel('X axis')
    plt.show()


def show_histogram_plot():
    """
    频率分布的图形表示
    :return:
    """
    a = np.array([22, 87, 5, 43, 56, 73, 55, 54, 11, 20, 51, 5, 79, 31, 27])
    np.histogram(a, bins=[0, 20, 40, 60, 80, 100])
    hist, bins = np.histogram(a, bins=[0, 20, 40, 60, 80, 100])
    print (hist)
    print (bins)

    a = np.array([22, 87, 5, 43, 56, 73, 55, 54, 11, 20, 51, 5, 79, 31, 27])
    plt.hist(a, bins=[0, 20, 40, 60, 80, 100])
    plt.title("histogram")
    plt.show()


def show_subplot_plot():
    np.random.seed(19680801)
    data = np.random.randn(2, 100)

    fig, axs = plt.subplots(2, 2, figsize=(5, 5))
    axs[0, 0].hist(data[0])
    axs[1, 0].scatter(data[0], data[1])
    axs[0, 1].plot(data[0], data[1])
    axs[1, 1].hist2d(data[0], data[1])

    plt.show()


def show_3d_plot():
    plt.rcParams['legend.fontsize'] = 10

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
    z = np.linspace(-2, 2, 100)
    r = z ** 2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    ax.plot(x, y, z, label='parametric curve')
    ax.legend()

    plt.show()


show_font_plot()
