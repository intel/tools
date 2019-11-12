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

import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.platform import gfile
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference
from tensorflow.tools.graph_transforms import TransformGraph
from google.protobuf import text_format

from api.quantize_graph import GraphRewriter

import os
import logging

logging.getLogger().setLevel(level=logging.INFO)


class GraphConverter:
    def __init__(self, input_graph, output_graph, inputs=[], outputs=[], per_channel=False, input_graph_is_binary=True):
        """Convert graph.

        :param input_graph: input graph pb file.
        :param output_graph: output graph pb file. If set, output directory should be exist.
        :param inputs: input nodes' names.
        :param outputs: output nodes' names.
        :param per_channel: if set True, enables weight quantization channel-wise.
        """
        self.input_graph = input_graph
        self.input_graph_binary_flag = input_graph_is_binary
        self.output_graph = output_graph
        self.inputs = inputs
        self.outputs = outputs
        self.per_channel = per_channel
        self.gen_calib_data_cmds = None
        self._low_precision_mode = 'eightbit'
        self._output_path = os.path.dirname(os.path.realpath(self.output_graph
                                                             if self.output_graph else self.input_graph))
        # generated graph files
        self._fp32_optimized_graph = None
        self._int8_dynamic_range_graph = None
        self._int8_logged_graph = os.path.join(self._output_path, 'int8_logged_graph.pb')
        self._requant_min_max_log = os.path.join(self._output_path, 'requant_min_max_log.txt')
        self._int8_frozen_range_graph = None
        self._check_args()

    def _check_args(self):
        if self.output_graph and not os.path.exists(os.path.dirname(self.output_graph)):
            raise ValueError('"output_graph" directory should be exist.')

    def convert(self):
        """Do convert, including:
            1) optimize fp32_frozen_graph,
            2) quantize graph,
            3) calibration,
            4) fuse RequantizeOp with fused quantized conv, and so on.

        :return:
        """
        if not self.gen_calib_data_cmds:
            raise ValueError('Pass an inference script to "gen_calib_data_cmds" to generate calibration data.')
        try:
            self._optimize_frozen_fp32_graph()
            self._quantize_graph()
            self._insert_logging()
            self._generate_calibration_data()
            self._freeze_requantization_ranges()
            self._fuse_requantize_with_fused_quantized_conv()
            self._post_clean()
        except Exception as e:
            logging.error('Failed to convert due to: %s', str(e))

    def _optimize_frozen_fp32_graph(self):
        """Optimize fp32 frozen graph."""

        self._fp32_optimized_graph = optimize_for_inference(self._read_graph(self.input_graph,
                                                                             self.input_graph_binary_flag),
                                                            self.inputs, self.outputs,
                                                            dtypes.float32.as_datatype_enum, False)

    def _quantize_graph(self):
        """quantize graph."""

        g = ops.Graph()
        with g.as_default():
            importer.import_graph_def(self._fp32_optimized_graph)

        rewriter = GraphRewriter(input_graph=self._fp32_optimized_graph,
                                 mode=self._low_precision_mode,
                                 quantized_input_range=None,
                                 intel_cpu_eightbitize=True,
                                 per_channel=self.per_channel)
        self._int8_dynamic_range_graph = rewriter.rewrite(self.outputs)

    def _transform_graph(self, in_graph, out_graph, transforms):
        """Transforms input graph.

        :param in_graph: input graph file or graphDef.
        :param out_graph: output graph file or graphDef.
        :param transforms: list of transforms.

        :return:
        """
        in_graph_def = in_graph if isinstance(in_graph, tf.compat.v1.GraphDef) else self._read_graph(in_graph)
        out_graph_def = TransformGraph(in_graph_def, self.inputs, self.outputs, transforms)
        if out_graph and not isinstance(out_graph, tf.compat.v1.GraphDef):
            f = gfile.GFile(out_graph, 'wb')
            f.write(out_graph_def.SerializeToString())
        return out_graph_def

    def _insert_logging(self):
        transforms = [
            'insert_logging(op=RequantizationRange{}, show_name=true, message="__requant_min_max:")'.format(
                "PerChannel" if self.per_channel else ""),
            'insert_logging(op=Min, show_name=true, message="__min:")',
            'insert_logging(op=Max, show_name=true, message="__max:")']
        
        self._transform_graph(self._int8_dynamic_range_graph, self._int8_logged_graph, transforms)

    def _generate_calibration_data(self):
        cmd = 'python ' + self.gen_calib_data_cmds
        cmd = cmd.format(self._int8_logged_graph)
        cmd += ' 2>&1 | tee {}'.format(self._requant_min_max_log)
        os.system(cmd)

    def _freeze_requantization_ranges(self):
        transforms = ['freeze_requantization_ranges(min_max_log_file="{}")'.format(self._requant_min_max_log),
                      'freeze_min(min_max_log_file="{}")'.format(self._requant_min_max_log),
                      'freeze_max(min_max_log_file="{}")'.format(self._requant_min_max_log)]
        self._int8_frozen_range_graph = self._transform_graph(self._int8_dynamic_range_graph,
                                                              self._int8_frozen_range_graph,
                                                              transforms)

    def _fuse_requantize_with_fused_quantized_conv(self):
        if not self.output_graph:
            self.output_graph = os.path.join(os.path.dirname(os.path.realpath(self.input_graph)),
                                             'int8_final_fused_graph.pb')
        transforms = ['fuse_quantized_conv_and_requantize', 'strip_unused_nodes']
        self._transform_graph(self._int8_frozen_range_graph, self.output_graph, transforms)
        logging.info('Converted graph file is saved to: %s', self.output_graph)

    def _read_graph(self, in_graph, in_graph_is_binary=True):
        """Reads input graph pb file as GraphDef.

        :param in_graph: input graph file.
        :return:
        """
        if not gfile.Exists(in_graph):
            logging.error('Input graph pb file %s does not exist.', self.input_graph)
            exit(-1)

        input_graph_def = graph_pb2.GraphDef()
        mode = "rb" if in_graph_is_binary else "r"
        with gfile.Open(in_graph, mode) as f:
            data = f.read()
            if in_graph_is_binary:
                input_graph_def.ParseFromString(data)
            else:
                text_format.Merge(data, input_graph_def)

        return input_graph_def

    def _post_clean(self):
        """Delete the temporarily files generated during the quantization process.

        :return: None 
        """
        if self._int8_logged_graph:
            os.remove(self._int8_logged_graph)

        if self._requant_min_max_log:
            os.remove(self._requant_min_max_log)
