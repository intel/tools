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

FROM intelaipg/intel-optimized-tensorflow:nightly-latestprs-bdw as binary_build

# Using Last commit from Feb 25th, 2019 as latest master is broken
# for building quantization tools.
ARG TF_BUILD_VERSION=cc624f49d27e3a66ad6243b8657756ee9bacb8d1

ENV WORKSPACE="/workspace"
ENV TF_WORKSPACE="${WORKSPACE}/tensorflow"
WORKDIR ${TF_WORKSPACE}

RUN git clone https://github.com/tensorflow/tensorflow.git . && \
    git checkout ${TF_BUILD_VERSION}

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
