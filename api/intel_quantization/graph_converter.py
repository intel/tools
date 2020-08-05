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
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util
# from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference

# from intel_quantization.quantize_graph import GraphRewriter
from .transform_graph.strip_unused import StripUnusedNodes
from .transform_graph.fold_batch_norm import FoldBatchNormNodes
from .transform_graph.insert_logging import InsertLogging
from .transform_graph.freeze_max_min import freeze_max
from .transform_graph.freeze_max_min import freeze_min
from .transform_graph.freeze_max_min import freeze_requantization_range
from .transform_graph.freeze_max_min import get_all_fp32_data, get_tensor_histogram, combine_histogram
from .transform_graph.fuse_quantized_conv_and_requantize import fuse_quantized_conv_and_requantize
from .transform_graph.fuse_column_wise_mul import FuseColumnWiseMul
from .transform_graph.rerange_quantized_concat import RerangeQuantizedConcat
from .util import read_graph, write_graph
from .quantize_graph.quantize_graph_for_intel_cpu import QuantizeGraphForIntel
from .quantize_graph.quantize_graph_common import QuantizeGraphHelper
import os
import shlex
import subprocess
import logging

logging.getLogger().setLevel(level=logging.INFO)

tf.compat.v1.disable_eager_execution()

TF_SUPPORTED_MAX_VERSION = '2.1.0'
TF_SUPPORTED_MIN_VERSION = '1.14.0'


class GraphConverter:
    def __init__(self, input_graph, output_graph, inputs=[], outputs=[], excluded_ops=[], excluded_nodes=[],
                 per_channel=False, input_graph_is_binary=True, algo='DIRECT'):
        """Convert graph.

        :param input_graph: input graph pb file.
        :param output_graph: output graph pb file. If set, output directory should be exist.
        :param inputs: input nodes' names.
        :param outputs: output nodes' names.
        :param excluded_ops: list of operations to be excluded from quantization.
        :param excluded_nodes: list of nodes to be excluded from quantization.
        :param per_channel: if set True, enables weight quantization channel-wise.
        :param input_graph_is_binary: default True, whether input graph is binary.
        """
        self.input_graph = input_graph
        self.input_graph_binary_flag = input_graph_is_binary
        self.output_graph = output_graph
        self.inputs = inputs
        self.outputs = outputs
        # quantize specific config
        self.per_channel = per_channel
        self.excluded_ops = excluded_ops
        self.excluded_nodes = excluded_nodes
        self.algo = algo
        self._low_precision_mode = 'eightbit'
        self._calibration_data = []
        self._fp32_print_data = []
        self.gen_calib_data_cmds = None
        self.debug = False
        self._check_tf_version()
        self._check_args()
        self._gen_tmp_filenames()
        self._kl_op_dict = {}
        self._kl_keys = []
        self._print_node_mapping = {}

    def _check_tf_version(self):
        is_supported_version = False
        try:
            from tensorflow import python
            if (hasattr(python, "pywrap_tensorflow") and hasattr(python.pywrap_tensorflow, "IsMklEnabled")):
                from tensorflow.python.pywrap_tensorflow import IsMklEnabled
            else:
                from tensorflow.python._pywrap_util_port import IsMklEnabled
            if IsMklEnabled() and (TF_SUPPORTED_MIN_VERSION <= tf.__version__ <= TF_SUPPORTED_MAX_VERSION):
                is_supported_version = True
        except Exception as e:
            raise ValueError(e)
        finally:
            if not is_supported_version:
                raise ValueError(str('Please install Intel® Optimizations for TensorFlow'
                                     ' or MKL enabled source build TensorFlow'
                                     ' with version >={} and <={}').format(TF_SUPPORTED_MIN_VERSION,
                                                                           TF_SUPPORTED_MAX_VERSION))

    def _check_args(self):
        if not gfile.Exists(self.input_graph):
            raise ValueError('Input graph pb file %s does not exist.' % self.input_graph)
        if self.output_graph and not os.path.exists(os.path.dirname(self.output_graph)):
            raise ValueError('"output_graph" directory does not exist.')

        self._output_path = os.path.dirname(os.path.realpath(self.output_graph if self.output_graph
                                                             else self.input_graph))

    def _gen_tmp_filenames(self):
        self._fp32_optimized_graph = os.path.join(self._output_path, 'fp32_optimized_graph.pb')
        self._int8_dynamic_range_graph = os.path.join(self._output_path, 'int8_dynamic_range_graph.pb')
        self._int8_logged_graph = os.path.join(self._output_path, 'int8_logged_graph.pb')
        self._fp32_logged_graph = os.path.join(self._output_path, 'fp32_logged_graph.pb')
        self._int8_frozen_range_graph = os.path.join(self._output_path, 'int8_frozen_range_graph.pb')
        if not self.output_graph:
            self.output_graph = os.path.join(self._output_path, 'int8_final_fused_graph.pb')
        # to keep temp graphDef
        self._tmp_graph_def = None

    def convert(self):
        """Do convert, including:
            1) optimize fp32_frozen_graph,
            2) quantize graph,
            3) calibration,
            4) fuse RequantizeOp with fused quantized conv, and so on.

        :return:
        """
        try:
            self._optimize_frozen_fp32_graph()
        except Exception as e:
            logging.error('Failed to optimize fp32 graph due to: %s', str(e))
            raise ValueError(e) from e
        else:
            self.quantize()

    def _get_fp32_print_node_names(self):
        offset_map = {
            "QuantizedConv2DWithBiasSumAndRelu": 3,
            "QuantizedConv2DWithBiasAndRelu": 2,
            "QuantizedConv2DWithBias": 1,
        }
        target_conv_op = []
        sorted_graph = QuantizeGraphHelper().get_sorted_graph(
            self._fp32_origin_graph, self.outputs)

        node_name_mapping = {
            node.name: node
            for node in self._tmp_graph_def.node if node.op != "Const"
        }

        for node in self._tmp_graph_def.node:
            if node.op in offset_map:
                target_conv_op.append(node.name.split('_eightbit_')[0])
        fp32_node_name_mapping = {
            node.name: node
            for node in sorted_graph.node if node.op != "Const"
        }
        sorted_node_names = [i.name for i in sorted_graph.node if i.op != "Const"]

        output_node_names = []
        for i in target_conv_op:
            if node_name_mapping[
                    i + "_eightbit_quantized_conv"].op == 'QuantizedConv2DWithBiasSumAndRelu':
                start_index = sorted_node_names.index(i)
                for index, value in enumerate(sorted_node_names[start_index:]):
                    if fp32_node_name_mapping[value].op.startswith(
                            "Add") and fp32_node_name_mapping[
                                sorted_node_names[start_index + index + 1]].op == "Relu":
                        output_node_names.append(
                            sorted_node_names[start_index + index + 1])
                        self._print_node_mapping[sorted_node_names[start_index + index + 1]] = i
            elif i in sorted_node_names:
                start_index = sorted_node_names.index(i)
                end_index = start_index + offset_map[node_name_mapping[
                    i + "_eightbit_quantized_conv"].op]
                output_node_names.append(sorted_node_names[end_index])
                self._print_node_mapping[sorted_node_names[end_index]] = i

        for i in output_node_names:
            self._kl_keys.append(';' + i + '__print__;__KL')

        InsertLogging(self._fp32_origin_graph,
                      node_name_list=output_node_names,
                      message="__KL:",
                      summarize=-1, dump_fp32=True).do_transformation()
        write_graph(self._fp32_origin_graph, self._fp32_logged_graph)

    def quantize(self):
        """Quantize graph only (without optimizing fp32 graph), including:
            1) quantize graph,
            2) calibration,
            3) fuse RequantizeOp with fused quantized conv, and so on.

        :return:
        """
        if not self.gen_calib_data_cmds:
            raise ValueError('Pass an inference command for accuracy to "gen_calib_data_cmds" '
                             'to generate calibration data.')
        try:
            self._quantize_graph()
            if self.algo == "KL":
                self._get_fp32_print_node_names()
                self._generate_calibration_data(self._fp32_logged_graph,
                                                self._fp32_print_data, True)

            self._insert_logging()
            self._generate_calibration_data(self._int8_logged_graph, self._calibration_data)
            self._freeze_requantization_ranges(self._kl_op_dict)
            self._fuse_requantize_with_fused_quantized_conv()
        except Exception as e:
            logging.error('Failed to quantize graph due to: %s', str(e))
            raise ValueError(e) from e
        finally:
            if not self.debug:
                self._post_clean()

    def _optimize_frozen_fp32_graph(self):
        """Optimize fp32 frozen graph."""

        self._tmp_graph_def = read_graph(self.input_graph, self.input_graph_binary_flag)
        dtypes = self._get_dtypes(self._tmp_graph_def)
        # self._tmp_graph_def = optimize_for_inference(self._tmp_graph_def, self.inputs, self.outputs, dtypes, False)
        self._tmp_graph_def = FuseColumnWiseMul(self._tmp_graph_def).do_transformation()
        self._tmp_graph_def = StripUnusedNodes(self._tmp_graph_def, self.inputs, self.outputs, dtypes).do_transform()
        self._tmp_graph_def = graph_util.remove_training_nodes(self._tmp_graph_def, self.outputs)
        self._tmp_graph_def = FoldBatchNormNodes(self._tmp_graph_def).do_transform()
        write_graph(self._tmp_graph_def, self._fp32_optimized_graph)
        self._fp32_origin_graph = self._tmp_graph_def

    def _quantize_graph(self):
        """quantize graph."""

        if not self._tmp_graph_def:
            self._tmp_graph_def = read_graph(self.input_graph, self.input_graph_binary_flag)

        g = ops.Graph()
        with g.as_default():
            importer.import_graph_def(self._tmp_graph_def)

        intel_quantizer = QuantizeGraphForIntel(self._tmp_graph_def,
                                                self.outputs, self.per_channel,
                                                excluded_ops=self.excluded_ops,
                                                excluded_nodes=self.excluded_nodes)
        self._tmp_graph_def = intel_quantizer.do_transform()

        if self.debug:
            write_graph(self._tmp_graph_def, self._int8_dynamic_range_graph)

    def _insert_logging(self):
        int8_dynamic_range_graph_def = graph_pb2.GraphDef()
        int8_dynamic_range_graph_def.CopyFrom(self._tmp_graph_def)
        InsertLogging(self._tmp_graph_def,
                      ops=["RequantizationRange{}".format("PerChannel" if self.per_channel else "")],
                      message="__requant_min_max:").do_transformation()
        InsertLogging(self._tmp_graph_def, ops=["Min"], message="__min:").do_transformation()
        InsertLogging(self._tmp_graph_def, ops=["Max"],
                      message="__max:").do_transformation()
        # InsertLogging(
        #     self._tmp_graph_def,
        #     ops=["QuantizedConv2DWithBiasAndRelu",
        #     "QuantizedConv2DWithBias"
        #     ],
        #     message="__KL:",
        #     summarize=-1).do_transformation()

        write_graph(self._tmp_graph_def, self._int8_logged_graph)
        self._tmp_graph_def.CopyFrom(int8_dynamic_range_graph_def)

    def _generate_calibration_data(self, graph, output, enable_kl_algo=False):
        cmd = self.gen_calib_data_cmds
        cmd = cmd.format(graph)
        p = subprocess.Popen(shlex.split(cmd),
                             stderr=subprocess.STDOUT,
                             stdout=subprocess.PIPE)
        while p.poll() is None:
            line = p.stdout.readline().strip().decode()
            if line and line.startswith(';'):
                if not enable_kl_algo:
                    output.append(line)

                if enable_kl_algo and line.rsplit(':')[0] in self._kl_keys:
                    fp32_data = get_all_fp32_data(line.rsplit(':')[-1])
                    key = self._print_node_mapping[line[1:].split('__print')[0]] + '_eightbit_requant_range'
                    if key not in self._kl_op_dict:
                        self._kl_op_dict[key] = get_tensor_histogram(fp32_data)
                    else:
                        self._kl_op_dict[key] = combine_histogram(self._kl_op_dict[key], fp32_data)

    def _freeze_requantization_ranges(self, additional_data=None):
        use_moving_average = self.algo == "MA"
        self._tmp_graph_def = freeze_max(self._tmp_graph_def,
                                         self._calibration_data,
                                         use_moving_average)
        self._tmp_graph_def = freeze_min(self._tmp_graph_def,
                                         self._calibration_data,
                                         use_moving_average)
        self._tmp_graph_def = freeze_requantization_range(
            self._tmp_graph_def, self._calibration_data, use_moving_average,
            additional_data)
        if self.debug:
            write_graph(self._tmp_graph_def, self._int8_frozen_range_graph)

    def _fuse_requantize_with_fused_quantized_conv(self):
        self._tmp_graph_def = fuse_quantized_conv_and_requantize(self._tmp_graph_def)
        # strip_unused_nodes with optimize_for_inference
        dtypes = self._get_dtypes(self._tmp_graph_def)
        # self._tmp_graph_def = optimize_for_inference(self._tmp_graph_def, self.inputs, self.outputs, dtypes, False)
        self._tmp_graph_def = StripUnusedNodes(self._tmp_graph_def, self.inputs, self.outputs, dtypes).do_transform()
        self._tmp_graph_def = graph_util.remove_training_nodes(self._tmp_graph_def, self.outputs)
        self._tmp_graph_def = FoldBatchNormNodes(self._tmp_graph_def).do_transform()
        RerangeQuantizedConcat(self._tmp_graph_def).do_transformation()
        write_graph(self._tmp_graph_def, self.output_graph)
        logging.info('Converted graph file is saved to: %s', self.output_graph)

    def _get_dtypes(self, in_graph_def):
        # TODO: keep dtypes list order as input list?
        dtypes = []
        for n in in_graph_def.node:
            if n.name in self.inputs:
                dtypes.append(n.attr["dtype"].type)

        return dtypes

    def _post_clean(self):
        """Delete the temporarily files generated during the quantization process.

        :return: None
        """
        if gfile.Exists(self._int8_logged_graph):
            os.remove(self._int8_logged_graph)
