# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/3/24 4:31 PM
# @File: setup
# @Email: mlshenkai@163.com

from distutils.core import setup
from setuptools import find_packages
with open("README.md", "r") as f:
    long_description = f.read()


setup(
    name="EasyAIScaffolding",
    version="1.0.0",
    description="A PyTorch-based AI development scaffolding",
    long_description=long_description,
    author="watcher shen",
    author_email="watcher.shen@gmail.com",
    url="https://github.com/mlshenkai/EasyAIScaffolding",
    license="MIT License",
    packages=find_packages(),
    platforms=["all"],
    classifiers=[]
)