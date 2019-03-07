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

#!/usr/bin/env bash

cd ../
TF_REPO=$(pwd)
OUTPUT=${TF_REPO}/output
mkdir ${OUTPUT}
cd ${TF_REPO}

# Build and run the docker image
QUANTIZATION_TAG="quantization:latest"
echo "Building..."
echo "Quantization container with tag: ${QUANTIZATION_TAG}."

docker build -f Dockerfile \
-t  ${QUANTIZATION_TAG} \
--build-arg HTTP_PROXY=${HTTP_PROXY} \
--build-arg HTTPS_PROXY=${HTTPS_PROXY} \
--build-arg http_proxy=${http_proxy} \
--build-arg https_proxy=${https_proxy} .

echo "Running container: ${QUANTIZATION_TAG}"
python launch_quantization.py \
--docker-image ${QUANTIZATION_TAG} \
--pre-trained-model-dir ${OUTPUT} \
--test

# clean up the output directory after the test is successfully done.
rm -rf ${OUTPUT}
