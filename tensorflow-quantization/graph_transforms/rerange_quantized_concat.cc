/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <algorithm>
#include <float.h>

#include "tensorflow/core/common_runtime/constant_folding.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/subgraph.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/tools/graph_transforms/fold_constants_lib.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace graph_transforms {

bool CollectConcatInputs(std::map<string, int> &offset_map,
                         std::map<string, const NodeDef *> &node_map,
                         std::vector<NodeDef *> &quantized_conv_nodes,
                         NodeDef *node) {
  string op_name = node->op();
  if (op_name.compare("QuantizedConcatV2") == 0) {
    NodeDef *concat_node = node;
    int n_input;
    GetNodeAttr(*concat_node, "N", &n_input);

    bool can_rerange = true;
    std::vector<NodeDef *> node_list;
    for (int i = 0; i < n_input; i++) {
      NodeDef *input_node = const_cast<NodeDef *>(
          node_map[NodeNameFromInput(concat_node->input(i))]);
      string input_op_type = input_node->op();
      if (offset_map.find(input_op_type) != offset_map.end()) {
        node_list.push_back(input_node);
      } else if (input_op_type.compare("QuantizedMaxPool") == 0 ||
                 input_op_type.compare("QuantizedAvgPool") == 0) {
        NodeDef *another_concat = const_cast<NodeDef *>(
            node_map[NodeNameFromInput(input_node->input(0))]);
        if (!CollectConcatInputs(offset_map, node_map, node_list,
                                 another_concat)) {
          std::cout << "Cannot rerange; for Concat which is input of Avg/Max "
                       "pooling: "
                    << another_concat->op() << "\n";
          can_rerange = false;
          break;
        }
      }
      // input of concat can be another concat
      else if (input_op_type.compare("QuantizedConcatV2") == 0) {
        if (!CollectConcatInputs(offset_map, node_map, node_list, input_node)) {
          std::cout << "Through concat Op, Cannot rerange the op: "
                    << input_node->op() << "\n";
          can_rerange = false;
          break;
        }
      } else {
        std::cout << "Cannot rerange ConcatOp inputs: invalid input op is "
                  << input_op_type
                  << " and name= " << input_node->name().c_str() << "\n";
        can_rerange = false;
        break;
      }
    }

    if (can_rerange) {
      for (auto a_node : node_list) {
        quantized_conv_nodes.push_back(a_node);
        std::cout << " Can Rerange: input of concat node "
                  << a_node->name().c_str() << " ; concat node is  "
                  << node->name().c_str() << std::endl;
      }
    }

    return can_rerange;
  }
  // else if (Quan..conv2d..) which is a input to Quant..Pool, push it to the
  // list and can rarange = true and return
  else if (op_name.compare("QuantizedConv2DWithBiasAndReluAndRequantize") ==
           0) {
    NodeDef *conv_node = node;
    bool can_rerange = true;

    quantized_conv_nodes.push_back(conv_node);
    std::cout
        << "adding Quantized..conv.. nodes as input of Quant..Pool. Nodename: "
        << conv_node->name().c_str() << std::endl;

    return can_rerange;
  } else {
    std::cout << "can not support " << op_name << "\n";
    return false;
  }
}

Status RerangeQuantizedConcat(const GraphDef &input_graph_def,
                              const TransformFuncContext &context,
                              GraphDef *output_graph_def) {
  *output_graph_def = input_graph_def;
  std::map<string, const NodeDef *> node_map;
  MapNamesToNodes(*output_graph_def, &node_map);

  std::map<string, int> offset_map;
  offset_map["QuantizedConv2DAndRequantize"] = 6;
  offset_map["QuantizedConv2DAndReluAndRequantize"] = 6;
  offset_map["QuantizedConv2DWithBiasAndRequantize"] = 7;
  offset_map["QuantizedConv2DWithBiasAndReluAndRequantize"] = 7;
  offset_map["QuantizedConv2DWithBiasSumAndReluAndRequantize"] = 7;
  offset_map["QuantizedConv2DWithBiasSignedSumAndReluAndRequantize"] = 7;

  std::set<NodeDef *> reranged_concat;

  for (auto &node_pair : node_map) {
    // std::string node_name = node_pair.first;
    NodeDef *node = (NodeDef *)node_pair.second;
    if (node->op().compare("QuantizedConcatV2") != 0) continue;

    NodeDef *concat_node = node;

    // check if all inputs are directly (or through pooling) from
    // QuantizedConv2D...AndRequantize
    // if they are not, do not do the rerange.
    std::vector<NodeDef *> quantized_conv_nodes;
    bool can_rerange = CollectConcatInputs(offset_map, node_map,
                                           quantized_conv_nodes, concat_node);

    if (!can_rerange) {
      printf("Can not re-range the inputs for node %s\n", node->name().c_str());
      continue;
    }

    reranged_concat.insert(concat_node);

    // find the combined range from all input tensors
    float combined_min = FLT_MAX, combined_max = -FLT_MAX;
    for (auto conv : quantized_conv_nodes) {
      int min_offset = offset_map[conv->op()];
      const NodeDef *min_freezed_output_node =
          node_map[NodeNameFromInput(conv->input(min_offset))];
      const NodeDef *max_freezed_output_node =
          node_map[NodeNameFromInput(conv->input(min_offset + 1))];

      const float min_freezed_output =
          GetNodeTensorAttr(*min_freezed_output_node, "value").flat<float>()(0);
      const float max_freezed_output =
          GetNodeTensorAttr(*max_freezed_output_node, "value").flat<float>()(0);

      if (min_freezed_output < combined_min) combined_min = min_freezed_output;
      if (max_freezed_output > combined_max) combined_max = max_freezed_output;
    }

    // set the combined range to quantized convolution node
    Tensor min_tensor(DT_FLOAT, {});
    min_tensor.flat<float>()(0) = combined_min;
    Tensor max_tensor(DT_FLOAT, {});
    max_tensor.flat<float>()(0) = combined_max;

    for (auto conv : quantized_conv_nodes) {
      int min_offset = offset_map[conv->op()];
      NodeDef *min_freezed_output_node =
          (NodeDef *)node_map[NodeNameFromInput(conv->input(min_offset))];
      NodeDef *max_freezed_output_node =
          (NodeDef *)node_map[NodeNameFromInput(conv->input(min_offset + 1))];
      SetNodeTensorAttr<float>("value", min_tensor, min_freezed_output_node);
      SetNodeTensorAttr<float>("value", max_tensor, max_freezed_output_node);
    }
  }

  // Convert the bias of QuantizedConv node from FP32 to INT32.
  // The conversions should be done when quantized conv node and requantize
  // got merged if the input to QuantizedConv2D...AndRequantize is another
  // QuantizedConv2D...AndRequantize.
  // However, after concat gets re-ranged, we can convert the case when the
  // input to
  // QuantizedConv2D...AndRequantize is another QuantizedConv2D...AndRequantize
  // through concat
  std::set<std::string> fused_requantized_bias_ops = {
      "QuantizedConv2DWithBiasAndRequantize",
      "QuantizedConv2DWithBiasAndReluAndRequantize",
      "QuantizedConv2DWithBiasSumAndReluAndRequantize",
      "QuantizedConv2DWithBiasSignedSumAndReluAndRequantize"};
  for (auto &node_pair : node_map) {
    NodeDef *node = const_cast<NodeDef *>(node_pair.second);
    if (fused_requantized_bias_ops.find(node->op()) ==
        fused_requantized_bias_ops.end())
      continue;

    NodeDef *conv_node = node;

    // search for another conv2d node, ignore concat and pooling ops
    bool done = false;
    NodeDef *current_node = conv_node;
    NodeDef *another_conv_node = nullptr;
    while (!done) {
      current_node = const_cast<NodeDef *>(
          node_map[NodeNameFromInput(current_node->input(0))]);
      if (offset_map.find(current_node->op()) != offset_map.end()) {
        another_conv_node = current_node;
        done = true;
      } else if (current_node->op().compare("QuantizedConcatV2") == 0) {
        if (reranged_concat.find(current_node) == reranged_concat.end())
          done = true;
      } else if (current_node->op().compare("QuantizedMaxPool") != 0 &&
                 current_node->op().compare("QuantizedAvgPool") != 0)
        done = true;
    }

    if (another_conv_node == nullptr) {
      std::cout << "Can not find the input from another "
                   "QuantizedConv2D...AndRequantize node\n";
      std::cout << "Can not convert the bias of " << conv_node->name()
                << " to INT32\n";
      continue;
    }
    std::cout << "the op type of the found node is " << another_conv_node->op()
              << "\n";

    DataType bias_type;
    GetNodeAttr(*conv_node, "Tbias", &bias_type);
    if (bias_type == DT_QINT32) continue;

    // double check the bias type should be FP32
    if (bias_type != DT_FLOAT) {
      std::cout << "Error: the bias type is: " << bias_type << " not correct\n";
      continue;
    }

    NodeDef *bias_node =
        const_cast<NodeDef *>(node_map[NodeNameFromInput(conv_node->input(2))]);
    int min_offset = offset_map[another_conv_node->op()];
    NodeDef *min_freezed_output_node = const_cast<NodeDef *>(
        node_map[NodeNameFromInput(another_conv_node->input(min_offset))]);
    NodeDef *max_freezed_output_node = const_cast<NodeDef *>(
        node_map[NodeNameFromInput(another_conv_node->input(min_offset + 1))]);
    const NodeDef *min_filter_node =
        node_map[NodeNameFromInput(conv_node->input(5))];
    const NodeDef *max_filter_node =
        node_map[NodeNameFromInput(conv_node->input(6))];
    const float min_input =
        GetNodeTensorAttr(*min_freezed_output_node, "value").flat<float>()(0);
    const float max_input =
        GetNodeTensorAttr(*max_freezed_output_node, "value").flat<float>()(0);
    const float min_filter =
        GetNodeTensorAttr(*min_filter_node, "value").flat<float>()(0);
    const float max_filter =
        GetNodeTensorAttr(*max_filter_node, "value").flat<float>()(0);

    TensorProto float_tensor_proto = bias_node->attr().at("value").tensor();
    Tensor float_tensor;
    CHECK(float_tensor.FromProto(float_tensor_proto));
    CHECK_EQ(float_tensor.dtype(), DT_FLOAT);
    float *p_bias_float = float_tensor.flat<float>().data();

    Tensor int32_tensor = Tensor(DT_QINT32, float_tensor.shape());
    qint32 *p_bias_int32 = int32_tensor.flat<qint32>().data();

    float bias_scale =
        255.0 * 127.0 / (std::max(std::abs(max_input), std::abs(min_input)) *
                         std::max(std::abs(max_filter), std::abs(min_filter)));
    int64 nelems = float_tensor.NumElements();
    for (int64 n = 0; n < nelems; n++)
      p_bias_int32[n] = (int32_t)(p_bias_float[n] * bias_scale);

    bias_node->clear_attr();
    AttrValue attr_type;
    attr_type.set_type(int32_tensor.dtype());
    bias_node->mutable_attr()->insert({"dtype", attr_type});
    AttrValue attr_tensor;
    TensorProto *t = attr_tensor.mutable_tensor();
    int32_tensor.AsProtoTensorContent(t);
    bias_node->mutable_attr()->insert({"value", attr_tensor});
    SetNodeAttr("Tbias", DT_QINT32, const_cast<NodeDef *>(conv_node));
  }

  return Status::OK();
}

REGISTER_GRAPH_TRANSFORM("rerange_quantized_concat", RerangeQuantizedConcat);

}  // namespace graph_transforms
}  // namespace tensorflow
