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
#

FROM gcr.io/deeplearning-platform-release/tf-cpu.1-14 as binary_build

# Using "TF_BUILD_VERSION=r1.15" if you need tensorflow 1.15.0
ARG TF_BUILD_VERSION=r1.14
ARG BAZEL_VERSION=0.24.1

ENV WORKSPACE="/workspace"
ENV TF_WORKSPACE="${WORKSPACE}/tensorflow"
WORKDIR ${TF_WORKSPACE}

RUN git clone -b ${TF_BUILD_VERSION} --single-branch https://github.com/tensorflow/tensorflow.git .

# Update graph_transforms files with customizations
COPY graph_transforms/* ${TF_WORKSPACE}/tensorflow/tools/graph_transforms/

# Copy quantization scripts
COPY quantization ${TF_WORKSPACE}/tensorflow/tools/quantization

# upgrade bazel
RUN wget https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
RUN bash bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh

# Build Graph Transform Tool
RUN bazel build --strip=never --config=mkl \
    tensorflow/tools/graph_transforms:transform_graph

# Build Summarize Graph Tool
RUN bazel build --strip=never --config=mkl \
    tensorflow/tools/graph_transforms:summarize_graph

CMD ["/bin/bash"]
