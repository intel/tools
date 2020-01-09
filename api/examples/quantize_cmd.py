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
import sys
import argparse
from tensorflow.python.platform import app

import intel_quantization.graph_converter as converter


def main(_):

    print (args.inputs.split(','), args.outputs.split(','), args.output_graph)
    if not os.path.exists(args.input_graph):
        print ("{} doesn't exist!".format(args.input_graph))
        sys.exit(-1)

    if args.inputs:
        inputs = args.inputs.split(',')
    else:
        inputs = []

    if args.outputs:
        outputs = args.outputs.split(',')
    else:
        outputs = []

    if args.excluded_ops:
        excluded_ops = args.exclude_ops.split(',')
    else:
        excluded_ops = []

    if args.excluded_nodes:
        excluded_nodes = args.exclude_nodes.split(',')
    else:
        excluded_nodes = []

    qt = converter.GraphConverter(args.input_graph, args.output_graph,
                                  inputs, outputs,
                                  excluded_ops, excluded_nodes,
                                  args.per_channel)
    qt.debug = args.debug
    if 'input_graph=' in args.callback:
        prefix = args.callback.split('input_graph=')[0]
        postfix = ' '.join(args.callback.split('input_graph=')[-1].split(' ')[1:])
        callback_cmd = prefix + 'input_graph={} '+ postfix
    else:
        callback_cmd = args.callback
    qt.gen_calib_data_cmds = args.callback
    qt.convert()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--callback',
        type=str,
        default=None,
        help='The calibration callback command.')
    
    parser.add_argument(
        '--inputs',
        type=str,
        default=None,
        help='The input op names of the graph.')
    
    parser.add_argument(
        '--outputs',
        type=str,
        default=None,
        help='The output op names of the graph.')
    
    parser.add_argument(
        '--input_graph', type=str, default=None, help='The input fp32 graph.')
    
    parser.add_argument(
        '--output_graph', type=str, default=None, help='The quantized graph')
    
    parser.add_argument(
        '--per_channel',
        type=bool,
        default=False,
        help='Apply the per channel quantization or not.')

    parser.add_argument(
        '--excluded_ops',
        type=str,
        default=None,
        help='The ops that excluded from quantization.')

    parser.add_argument(
        '--excluded_nodes',
        type=str,
        default=None,
        help='The nodes that excluded from quantization.')
    
    parser.add_argument(
        '--debug', type=bool, default=False, help='Debug mode.')

    args = parser.parse_args()

    app.run()
