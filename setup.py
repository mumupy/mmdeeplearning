#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/22 10:14
# @Author  : ganliang
# @File    : setup.py.py
# @Desc    : 安装包

from setuptools import setup, find_packages

"""
编译 python setup.py build
安装 python setup.py install
打包（源代码发布） python setup.py sdist
将项目上传到pypi python setup.py sdist upload
打包成可执行（exe、rpm） python setup.py bdist
  --formats=rpm      RPM distribution
  --formats=gztar    gzip'ed tar file
  --formats=bztar    bzip2'ed tar file
  --formats=ztar     compressed tar file
  --formats=tar      tar file
  --formats=wininst  Windows executable installer
  --formats=zip      ZIP file 
"""
# setup(
#     name='mmscrapy',
#     version='1.0',
#     packages=find_packages(),
#     entry_points={'scrapy': ['settings = mmscrapy.settings']}
# )

setup(name="mmdeeplearning",
      version="0.0.1",
      description="""mmdeeplearning是一个学习深度学习框架的demo项目，通过该项目了解深度学习的基本理念和各个深度学习框架的使用方式和原理。目前主要包含如下几个主要功能模块：1、numpy 高纬度数据模型。 2、matplotlib 绘图组件。3、pandas。4、mxnet 深度学习框架。5、tensorflow 深度学习框架""",
      long_description=open('README.md', encoding="utf-8").read(),
      long_description_content_type="text/markdown",
      author="甘亮",
      author_email="lovercws@gmail.com",
      keywords="深度学习、python、numpy、mxnet",
      # py_modules=["main"], #将一个模块打成包
      packages=find_packages(),
      license='Apache License',
      include_package_data=True,
      platforms="any",
      url="https://github.com/mumupy/mmdeeplearning.git",
      install_requires=[],
      scripts=[],
      entry_points={'deeplearning': ['settings = mmdeeplearning.settings']}
      )
