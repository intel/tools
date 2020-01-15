#
#  -*- coding: utf-8 -*-
#
#  Copyright (c) 2019-2020 Intel Corporation
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
import json
from tensorflow.python.platform import app

import intel_quantization.graph_converter as converter


class ModelZooBridge:
    """The type definition in model.json config file
    """
    MODEL_NAME = 'MODEL_NAME'
    LAUNCH_BENCHMARK_PARAMS = 'LAUNCH_BENCHMARK_PARAMS'
    LAUNCH_BENCHMARK_SCRIPT = 'LAUNCH_BENCHMARK_SCRIPT'
    LAUNCH_BENCHMARK_CMD = 'LAUNCH_BENCHMARK_CMD'
    IN_GRAPH = 'IN_GRAPH'
    DATA_LOCATION = 'DATA_LOCATION'
    MODEL_SOURCE_DIR = 'MODEL_SOURCE_DIR'
    DIRECT_PASS_PARAMS_TO_MODEL = "DIRECT_PASS_PARAMS_TO_MODEL"

    QUANTIZE_GRAPH_CONVERTER_PARAMS = 'QUANTIZE_GRAPH_CONVERTER_PARAMS'
    INPUT_NODE_LIST = 'INPUT_NODE_LIST'
    OUTPUT_NODE_LIST = 'OUTPUT_NODE_LIST'
    EXCLUDED_OPS_LIST = 'EXCLUDED_OPS_LIST'
    EXCLUDED_NODE_LIST = 'EXCLUDED_NODE_LIST'
    PER_CHANNEL_FLAG = 'PER_CHANNEL_FLAG'

    def __init__(self,
                 model_name,
                 in_graph,
                 data_location,
                 models_zoo_path,
                 models_source_dir=None,
                 models_config_file=None):
        self.in_graph = in_graph
        self.data_location = data_location
        self.model_name = model_name
        self.models_zoo_path = models_zoo_path
        self.models_source_dir = models_source_dir
        self.models_config_file = models_config_file

        self.model_param_dict = {}
        self._supported_model_list = []
        self._models_parser()

    def _models_parser(self):
        if os.path.exists(self.models_config_file):
            with open(self.models_config_file, 'r') as config:
                config_object = json.load(config)
                for params in config_object:
                    if params[ModelZooBridge.
                              MODEL_NAME] not in self.supported_model_list:
                        self.supported_model_list.append(
                            params[ModelZooBridge.MODEL_NAME])
                    if self.model_name != params[ModelZooBridge.MODEL_NAME]:
                        continue
                    for key in params.keys():
                        self.model_param_dict[key] = params[key]
                        print("Model Config: %s:%s" %
                              (key, self.model_param_dict[key]))
        else:
            print("Warning: File {} does not exist.".format(
                self.models_config_file))
        print(
            "Model Config: Supported models - %s" % self.supported_model_list)

    def _inference_calib_cmd(self):
        launch_benchmark_params = self.model_param_dict[
            ModelZooBridge.LAUNCH_BENCHMARK_PARAMS]
        inference_calib_cmd = 'python ' + os.path.join(
            self.models_zoo_path,
            launch_benchmark_params[ModelZooBridge.LAUNCH_BENCHMARK_SCRIPT])
        inference_calib_cmd += ' ' + ' '.join(
            launch_benchmark_params[ModelZooBridge.LAUNCH_BENCHMARK_CMD])
        inference_calib_cmd += ' ' + launch_benchmark_params[ModelZooBridge.
                                                             DATA_LOCATION].format(
                                                                 self.
                                                                 data_location)
        inference_calib_cmd += ' ' + launch_benchmark_params[ModelZooBridge.
                                                             IN_GRAPH]
        if self.models_source_dir and ModelZooBridge.MODEL_SOURCE_DIR in launch_benchmark_params.keys(
        ):
            inference_calib_cmd += ' ' + launch_benchmark_params[ModelZooBridge.
                                                                 MODEL_SOURCE_DIR].format(
                                                                     self.
                                                                     models_source_dir
                                                                 )
        if ModelZooBridge.DIRECT_PASS_PARAMS_TO_MODEL in launch_benchmark_params.keys(
        ):
            inference_calib_cmd += ' ' + ' '.join(
                launch_benchmark_params[ModelZooBridge.
                                        DIRECT_PASS_PARAMS_TO_MODEL])
        print('Inference Calibration Command: %s' % inference_calib_cmd)
        return inference_calib_cmd

    def _quantize_params_dict(self):
        params = self.model_param_dict[
            ModelZooBridge.QUANTIZE_GRAPH_CONVERTER_PARAMS]
        return params

    def _supported_model_list(self):
        return self._supported_model_list

    inference_calib_cmd = property(_inference_calib_cmd)
    quantize_params_dict = property(_quantize_params_dict)
    supported_model_list = property(_supported_model_list)


def main(_):
    # Build the bridge with intel model zoo configuration for inference run
    config_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "../config/models.json")
    model = ModelZooBridge(args.model, args.in_graph, args.data_location,
                           args.models_zoo_location, args.models_source_dir,
                           config_file_path)

    # pick up the quantization calibration parameters from models.json
    inference_cmd_gen_minmax_log = model.inference_calib_cmd
    graph_converter_params = model.quantize_params_dict
    inputs = []
    outputs = []
    excluded_ops = []
    excluded_nodes = []
    per_channel = False
    if ModelZooBridge.INPUT_NODE_LIST in graph_converter_params.keys():
        inputs = graph_converter_params[ModelZooBridge.INPUT_NODE_LIST]
    if ModelZooBridge.OUTPUT_NODE_LIST in graph_converter_params.keys():
        outputs = graph_converter_params[ModelZooBridge.OUTPUT_NODE_LIST]
    if ModelZooBridge.EXCLUDED_OPS_LIST in graph_converter_params.keys():
        excluded_ops = graph_converter_params[ModelZooBridge.EXCLUDED_OPS_LIST]
    if ModelZooBridge.EXCLUDED_NODE_LIST in graph_converter_params.keys():
        excluded_nodes = graph_converter_params[
            ModelZooBridge.EXCLUDED_NODE_LIST]
    if ModelZooBridge.PER_CHANNEL_FLAG in graph_converter_params.keys():
        per_channel = graph_converter_params[ModelZooBridge.PER_CHANNEL_FLAG]

    # Call the GraphConverter to do the FP32 calibration, INT8 quantization, and INT8 calibration
    qt = converter.GraphConverter(args.in_graph, args.out_graph, inputs,
                                  outputs, excluded_ops, excluded_nodes,
                                  per_channel)
    qt.debug = args.debug
    qt.gen_calib_data_cmds = inference_cmd_gen_minmax_log
    qt.convert()


if __name__ == '__main__':
    required_arg = "--help" not in sys.argv

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='The model name to run quantization.')
    parser.add_argument(
        '--in_graph',
        type=str,
        default=None,
        required=required_arg,
        help='The path to input original fp32 frozen graph.')
    parser.add_argument(
        '--out_graph',
        type=str,
        default=None,
        help='The path to generated output int8 frozen graph.')
    parser.add_argument(
        '--data_location',
        type=str,
        default=None,
        help='The path to dataset with tfrecord format.')
    parser.add_argument(
        '--models_zoo_location',
        type=str,
        default=None,
        help='Specify the root path to Model Zoo for Intel® Architecture'
        ' to run the model from Model Zoo.')
    parser.add_argument(
        '--models_source_dir',
        type=str,
        default=None,
        help=
        'Specify the path to tensorflow models which are only required by few models in Model Zoo.'
        ' This argument can only be used in conjunction with a --models_zoo_location.'
    )
    parser.add_argument(
        "--debug",
        type=bool,
        default=False,
        help="Launches debug mode with rich debug information.")
    args = parser.parse_args()
    app.run()
