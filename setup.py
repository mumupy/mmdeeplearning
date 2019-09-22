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
      description="mmscrapy爬虫程序是使用scrapy框架搭建的爬虫项目，解scrapy的使用方式和学习scrapy的使用技巧、编写自己的爬虫程序、分布式爬虫功能支持，scrapy支持很多特性，不必要自己创轮子",
      long_description=open('README.md', encoding="utf-8").read(),
      long_description_content_type="text/markdown",
      author="甘亮",
      author_email="lovercws@gmail.com",
      keywords="python版本的爬虫程序",
      # py_modules=["main"], #将一个模块打成包
      packages=find_packages(),
      license='Apache License',
      include_package_data=True,
      platforms="any",
      url="https://github.com/mumupy/mmdeeplearning.git",
      install_requires=[],
      scripts=[],
      entry_points={'deeplearning': ['settings = mmscrapy.settings']}
      )
