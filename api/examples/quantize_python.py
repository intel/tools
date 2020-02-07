#
#  -*- coding: utf-8 -*-
#
#  Copyright (c) 2020 Intel Corporation
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

import os
import graph_converter as converter

_RN50_MODEL = os.path.join(
    os.environ['HOME'],
    'tools/api/models/resnet50/resnet50_fp32_pretrained_model.pb')
_DATA_LOC = '/lustre/dataset/tensorflow/imagenet'


def rn50_callback_cmds():
    script = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'models/resnet50/accuracy.py')
    # You can set up larger batch_size and num_batches to get better accuracy, more time is needed accordingly.
    # Leave `--input_graph={}` unformatted.
    flags = ' --batch_size=50' + \
            ' --num_inter_threads=2' + \
            ' --num_intra_threads=28' + \
            ' --input_graph={}' + \
            ' --data_location={}'.format(_DATA_LOC) + \
            ' --num_batches 10'
    return script + flags


if __name__ == '__main__':
    # ResNet50 v1.0 quantization example.
    rn50 = converter.GraphConverter(_RN50_MODEL, None, ['input'], ['predict'])
    # pass an inference script to `gen_calib_data_cmds` to generate calibration data.
    rn50.gen_calib_data_cmds = rn50_callback_cmds()
    rn50.convert()
