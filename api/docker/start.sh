#
# -*- coding: utf-8 -*-
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

QUANTIZE_MODEL=${WORKSPACE}/quantization/api/quantize_model.py

if [ -f ${QUANTIZE_MODEL} ]; then
    python ${QUANTIZE_MODEL} \
        --in_graph=${IN_GRAPH} \
        --model=${MODEL_NAME} \
        --out_graph=${OUT_GRAPH} \
        --data_location=${DATA_LOCATION} \
        --models_zoo=${MODELS_ZOO} \
        --models_source_dir=${MODELS_SOURCE_DIR} \
        --debug=${DEBUG} 
else
    echo "${QUANTIZE_MODEL} does not exit !"
    exit 1
fi
