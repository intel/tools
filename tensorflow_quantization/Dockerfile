#
# Copyright (c) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: EPL-2.0
#
#

FROM ubuntu:18.04

ARG TF_BUILD_VERSION=r1.14
ARG PYTHON="python"
ARG PYTHON3_DEV=""
ARG WHL_DIR="/tmp/pip"
ARG PIP="pip"

RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
        ${PYTHON} \
        ${PYTHON}-dev \
        ${PYTHON}-pip \
        ${PYTHON}-setuptools \
        ${PYTHON}-wheel \
        build-essential \
        curl \
        git \
        libcurl3-dev \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libpng-dev \
        libssl-dev \
        libzmq3-dev \
        openjdk-8-jdk \
        openjdk-8-jre-headless \
        pkg-config \
        software-properties-common \
        unzip \
        wget \
        zip \
        zlib1g-dev \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN ${PIP} --no-cache-dir install \
        Pillow \
        h5py \
        ipykernel \
        keras_applications \
        keras_preprocessing \
        mock \
        numpy \
        pandas \
        scipy \
        sklearn \
        && \
    ${PYTHON} -m ipykernel.kernelspec

# Set up Bazel
ENV BAZEL_VERSION 0.23.0
WORKDIR /workspace
RUN mkdir /bazel && \
    cd /bazel && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -o /bazel/LICENSE.txt https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE && \
    chmod +x bazel-*.sh && \
    ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    cd / && \
    rm -f /bazel/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh

# Download and build TensorFlow.
ENV TF_WORKSPACE /workspace/tensorflow
WORKDIR ${TF_WORKSPACE}

# Enable checking out both tags and branches
RUN export TAG_PREFIX="v" && \
    echo ${TF_BUILD_VERSION} | grep -q ^${TAG_PREFIX}; \
    if [ $? -eq 0 ]; then \
        git clone --depth=1 https://github.com/tensorflow/tensorflow.git . && \
        git fetch --tags && \
        git checkout ${TF_BUILD_VERSION}; \
   else \
        git clone --depth=1 --branch=${TF_BUILD_VERSION} https://github.com/tensorflow/tensorflow.git . ; \
    fi

RUN yes "" | ${PYTHON} configure.py
RUN cp .bazelrc /root/.bazelrc

ENV CI_BUILD_PYTHON ${PYTHON}

# Set bazel build parameters in .bazelrc in parameterized_docker_build.sh
# Use --copt=-march values to get optimized builds appropriate for the hardware
#   platform of your choice.
# For ivy-bridge or sandy-bridge
# --copt=-march="avx" \
# For haswell, broadwell, or skylake
# --copt=-march="avx2" \
RUN cp .bazelrc /root/.mkl.bazelrc
RUN echo "import /root/.mkl.bazelrc" >>/root/.bazelrc

RUN tensorflow/tools/ci_build/builds/configured CPU \
    bazel --bazelrc=/root/.bazelrc build -c opt \
    tensorflow/tools/pip_package:build_pip_package && \
    bazel-bin/tensorflow/tools/pip_package/build_pip_package "${WHL_DIR}" && \
    ${PIP} --no-cache-dir install --upgrade "${WHL_DIR}"/tensorflow-*.whl && \
    rm -rf /root/.cache
# Clean up Bazel cache when done.
RUN bazel clean --expunge --color=yes && \
    rm -rf /root/.cache

# Update graph_transforms files with customizations
COPY graph_transforms/* ${TF_WORKSPACE}/tensorflow/tools/graph_transforms/

# Copy quantization scripts
COPY quantization ${TF_WORKSPACE}/tensorflow/tools/quantization

# Build Graph Transform Tool
RUN bazel build --strip=never --config=mkl \
    tensorflow/tools/graph_transforms:transform_graph

# Build Summarize Graph Tool
RUN bazel build --strip=never --config=mkl \
    tensorflow/tools/graph_transforms:summarize_graph

CMD ["/bin/bash"]
