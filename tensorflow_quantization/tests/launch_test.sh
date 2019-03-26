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
set -e
set -x

cd ../
TF_REPO=$(pwd)
OUTPUT=${TF_REPO}/output
LOGS=${OUTPUT}/test_logs.txt

# OUTPUT directory exists when test fails,
# so we need to clean up and re-create new one for next test run.
if [ -d ${OUTPUT} ]
then
    rm -rf ${OUTPUT}
fi

mkdir ${OUTPUT}

if [ $? -eq 1 ]
then
    echo "Output directory creation for test scripts FAILED" | tee ${LOGS}
    exit 1
else
    echo "Created output directory for running test scripts at: ${OUTPUT}" | tee ${LOGS}
fi

cd ${TF_REPO}

# Build and run the docker image
QUANTIZATION_TAG="quantization:latest"
echo "Building quantization tools docker image with tag: ${QUANTIZATION_TAG}" | tee -a ${LOGS}

docker build -f Dockerfile \
-t  ${QUANTIZATION_TAG} \
--build-arg HTTP_PROXY=${HTTP_PROXY} \
--build-arg HTTPS_PROXY=${HTTPS_PROXY} \
--build-arg http_proxy=${http_proxy} \
--build-arg https_proxy=${https_proxy} . | tee -a ${LOGS}

if [ "${PIPESTATUS[0]}" -eq "0" ]
then
    echo ""
    echo "******** Running Quantization Test Scripts ********" | tee -a ${LOGS}
    python launch_quantization.py \
    --docker-image ${QUANTIZATION_TAG} \
    --pre-trained-model-dir ${OUTPUT} \
    --verbose --test | tee -a ${LOGS}

    if [ "${PIPESTATUS[0]}" -ne 0 ] || [[ "`grep 'usage: bazel-bin/' ${LOGS} > /dev/null`" != "" ]]
    then
        echo "Test scripts run FAILED !!" | tee -a ${LOGS}
        echo "Please check logs at: ${LOGS}" | tee -a ${LOGS}
        exit 1
    else
        echo "Test scripts run completed SUCCESSFULLY !!" | tee -a ${LOGS}
        rm -rf ${OUTPUT}
    fi
else
    echo "Error: Quantization tools docker build FAILED " | tee -a ${LOGS}
    echo "Test scripts haven't INITIATED, please fix issue and re-run" | tee -a ${LOGS}
    echo "Please check logs at: ${LOGS}" | tee -a ${LOGS}
    exit 1
fi
