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
# usage example:
# 1. command line:
#    python summarize_graph.py --in_graph=path_to_graph
# 2. API :
#   inputs, outputs = SummarizeGraph(in_graph)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging

from tensorflow.core.framework import graph_pb2
from tensorflow.python.platform import gfile
from tensorflow.core.framework import types_pb2 as types
from google.protobuf import text_format

FORMAT = "%(asctime)s %(filename)s %(levelname)s %(message)s"
logging.basicConfig(level=logging.INFO,
                    datefmt="%a, %d %b %Y %H:%M:%S",
                    format=FORMAT)

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--in_graph", default="", help="Path to graph")
parser.add_argument("-b", "--input_binary", action="store_true", help="For binary graph only")
args = parser.parse_args()

DTYPEDICT = {types.DT_FLOAT: "float",
             types.DT_INVALID: "INVALID",
             types.DT_DOUBLE: "double",
             types.DT_INT32: "int32",
             types.DT_UINT32: "uint32",
             types.DT_UINT8: "uint8",
             types.DT_UINT16: "uint16",
             types.DT_INT16: "int16",
             types.DT_INT8: "int8",
             types.DT_STRING: "string",
             types.DT_COMPLEX64: "complex64",
             types.DT_COMPLEX128: "complex128",
             types.DT_INT64: "int64",
             types.DT_UINT64: "uint64",
             types.DT_BOOL: "bool",
             types.DT_QINT8: "qint8",
             types.DT_QUINT8: "quint8",
             types.DT_QUINT16: "quint16",
             types.DT_QINT16: "qint16",
             types.DT_QINT32: "qint32",
             types.DT_BFLOAT16: "bfloat16",
             types.DT_HALF: "half",
             types.DT_RESOURCE: "resource",
             types.DT_VARIANT: "variant"}


def data_type_string_internal(dtype):
    if dtype not in DTYPEDICT.keys():
        print("Unrecognized DataType enum value")
        return "unknown dtype enum" + "(" + str(dtype) + ")"
    return DTYPEDICT[dtype]


def print_node_info(node):
    shape_description = "None"
    dtype = types.DT_INVALID

    if 'shape' in node.attr:
        shape_proto = []
        for i, _ in enumerate(node.attr['shape'].shape.dim):
            shape_proto.append(node.attr['shape'].shape.dim[i].size)
        shape_description = shape_proto

    if 'dtype' in node.attr:
        dtype = node.attr['dtype'].type

    dtype_string = data_type_string_internal(dtype)
    node_info = "(name=" + node.name + ", type=" + dtype_string + \
                "(" + str(dtype) + ")" + ", shape=" + str(shape_description) + ")"
    print(node_info)
    return node.name


def node_name_from_input(input_name):
    inputs = input_name.split(":")

    if inputs[0][0] == "^":
        inputs[0] = inputs[0][len("^"):]
    return inputs[0]


def map_nodes_to_outputs(graph_def, result):
    for node in graph_def.node:
        for input_name in node.input:
            input_node_name = node_name_from_input(input_name)
            result.setdefault(input_node_name, []).append(node)


def summarize_graph(graph):
    in_nodes_name = []
    out_nodes_name = []
    placeholders = []
    variables = []
    op_counts = {}

    for node in graph.node:
        if node.op in op_counts.keys():
            op_counts[node.op] += 1
        else:
            op_counts[node.op] = 1

        if node.op == "Placeholder":
            placeholders.append(node)
        if node.op in ["Variable", "VariableV2"]:
            variables.append(node)

    if not placeholders:
        print("No input spotted.")
    else:
        print("Found ", len(placeholders), " possible inputs: ")
        for node in placeholders:
            in_nodes_name.append(print_node_info(node))

    output_map = {}
    outputs = []
    map_nodes_to_outputs(graph, output_map)
    unlikely_output_types = {'Const', 'Assign', 'NoOp', 'placeholder'}
    for node in graph.node:
        if node.name not in output_map.keys() and node.op not in unlikely_output_types:
            outputs.append(node)

    if not outputs:
        print("No outputs spotted.")
    else:
        print("Found ", len(outputs), " possible outputs: ")
        for node in outputs:
            output_info = "(name=" + node.name + ", op=" + node.op + ")"
            print(output_info)
            out_nodes_name.append(node.name)

    for function in graph.library.function:
        for node in function.node_def():
            if node.op in op_counts.keys():
                op_counts[node.op] += 1
            else:
                op_counts[node.op] = 1

    op_counts_vec = sorted(op_counts.items(), key=lambda item: item[1], reverse=True)
    print("Op types used: ")
    for v in op_counts_vec:
        print(v[1], "", v[0], ",", end="")
    print("")

    return in_nodes_name, out_nodes_name


def main():
    in_graph = args.in_graph

    if in_graph is None or not gfile.Exists(in_graph):
        logging.error("--in_graph %s does not exit" % in_graph)
        return

    graph = graph_pb2.GraphDef()
    mode = "rb" if args.input_binary else "r"
    with gfile.Open(in_graph, mode) as f:
        data = f.read()
        if args.input_binary:
            graph.ParseFromString(data)
        else:
            text_format.Merge(data, graph)

    summarize_graph(graph)


def SummarizeGraph(in_graph=None, input_binary=True):
    if not in_graph or not gfile.Exists(in_graph):
        raise ValueError("in_graph %s does not exist" % str(in_graph))

    graph = graph_pb2.GraphDef()
    mode = "rb" if input_binary else "r"
    with gfile.Open(in_graph, mode) as f:
        data = f.read()
        if input_binary:
            graph.ParseFromString(data)
        else:
            text_format.Merge(data, graph)

    return summarize_graph(graph)


if __name__ == "__main__":
    main()
