环境安装
========

python安装
>>>>>>>>>>

可以到 https://www.python.org/downloads 进行安装。

anaconda安装
>>>>>>>>>>>>

可以到 https://www.anaconda.com/distribution/#download-section 进行安装。

安装完成之后 进行环境准备。

::

    conda update conda
    conda update anaconda

    conda create -n python3.5.2 python=3.5.2
    activate python3.5.2

    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
    conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/main/
    conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/free/
    conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge/
    conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/msys2/
    conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/bioconda/
    conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/menpo/

    conda config --set show_channel_urls yes

pip国内加速
>>>>>>>>>>>

pip默认会访问国外pypi进行包下载，导致下载速度过慢，这里可以更改pip的源，加快访问速度。

::

    #配置清华PyPI镜像（如无法运行，将pip版本升级到>=10.0.0）
    pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple
    pip config set global.trusted-host https://mirrors.ustc.edu.cn/pypi/web/simple
    pip config list


jupyter安装
>>>>>>>>>>>

jupyter是一个在线文本编辑器，可以支持多种开发语言进行开发，如java、python、go等开发语言。

::

    pip install jupyter
    jupyter notebook
