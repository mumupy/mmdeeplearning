#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/24 21:43
# @Author  : ganliang
# @File    : latex.py
# @Desc    : TODO

latex_engine = 'xelatex'
latex_elements = {
    'papersize': 'a4paper',
    'pointsize': '11pt',
    'preamble': r'''
\usepackage{xeCJK}
\setCJKmainfont[BoldFont=STZhongsong, ItalicFont=STKaiti]{STSong}
\setCJKsansfont[BoldFont=STHeiti]{STXihei}
\setCJKmonofont{STFangsong}
\XeTeXlinebreaklocale "zh"
\XeTeXlinebreakskip = 0pt plus 1pt
\parindent 2em
\definecolor{VerbatimColor}{rgb}{0.95,0.95,0.95}
\setcounter{tocdepth}{3}
\renewcommand\familydefault{\ttdefault}
\renewcommand\CJKfamilydefault{\CJKrmdefault}
'''
}
# 设置文档
latex_documents = [
    ("", 'sphinx.tex', '你的第一本 Sphinx 书',
     '作者：qiwihui', 'manual', True),
]
