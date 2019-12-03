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

import os
import argparse

from tensorflow.python.platform import app

import graph_converter as converter


def rn50_callback_cmds(data_location):
    script = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models/resnet50/accuracy.py')
    # Set bigger value of batch_size and num_batches to get better accuracy, but more time is required.
    # Leave `--input_graph={}` unformatted for automatic fill in graph_coverter. 
    flags = ' --batch_size=50' + \
            ' --input_graph={}' + \
            ' --data_location={}'.format(data_location) + \
            ' --num_batches 10'
    return 'python ' + script + flags


def rn50v1_5_callback_cmds(data_location):
    script = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          'models/resnet50v1_5/eval_image_classifier_inference.py')
    flags = " --batch-size=50" + \
            " --input-graph={}" + \
            " --data-location={}".format(data_location) + \
            " --steps=10"
    return 'python ' + script + flags


def main(_):
    c = None
    if args.model == 'resnet50':
        c = converter.GraphConverter(args.model_location, None, ['input'], ['predict'])
        # This command is to execute the inference with small subset of the training dataset, and get the min and max log output.
        c.gen_calib_data_cmds = rn50_callback_cmds(args.data_location)
    elif args.model == 'resnet50_v1':
        c = converter.GraphConverter(args.model_location, None, ['input_tensor'], ['ArgMax', 'softmax_tensor'],
                                     per_channel=True)
        c.gen_calib_data_cmds = rn50v1_5_callback_cmds(args.data_location)
    c.convert()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet50', help='The model name')
    parser.add_argument('--model_location', type=str, default=None, help='The original fp32 frozen graph')
    parser.add_argument('--data_location', type=str, default=None, help='The dataset in tfrecord format')
    args = parser.parse_args()

    app.run()
