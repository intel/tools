#
#  -*- coding: utf-8 -*-
#
#  Copyright (c) 2019 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

from os import path
from setuptools import find_packages
from setuptools import setup

with open(path.join(path.dirname(path.realpath(__file__)), 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='intel-quantization',
    version='1.0b1',
    author='intel',
    description='The Python programming APIs packages for Intel® AI Quantization Tools for Tensorflow*.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/IntelAI/tools',
    packages=find_packages(exclude=['docker', 'config', 'models', 'tests', 'tools']),
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    keywords='Intel® AI Quantization Tools for Tensorflow*',
    python_requires='>=3.4, !=3.1.*, !=3.2.*, !=3.3.*, <3.8'
)
