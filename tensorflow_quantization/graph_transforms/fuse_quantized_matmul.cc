/* Copyright (c) 2019 Intel Corporation

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

Status FuseQuantizedMatMulAndRequantize(const GraphDef &input_graph_def,
                                        const TransformFuncContext &context,
                                        GraphDef *output_graph_def) {
  std::map<string, const NodeDef *> node_map;
  MapNamesToNodes(input_graph_def, &node_map);
  GraphDef replaced_graph_def;
  TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
      input_graph_def,  // clang-format off

      {"Requantize",
          {
              {"QuantizedMatMulWithBiasAndRelu"},
              {"QuantizedMatMulWithBiasAndRelu"},
              {"QuantizedMatMulWithBiasAndRelu"},
              {"Const"},
              {"Const"}
          }
      },
      [&node_map](const NodeMatch& match, const std::set<string>& input_nodes,
         const std::set<string>& output_nodes,
         std::vector<NodeDef>* new_nodes) {
        // Find all the nodes we expect in the subgraph.
        const NodeDef& requantize_node = match.node;
        CHECK_EQ("Requantize", requantize_node.op());
        const NodeDef& quantized_matmul_node = match.inputs[0].node;
        const NodeDef& const_requantize_range_min_node = match.inputs[3].node;
        CHECK_EQ("Const", const_requantize_range_min_node.op());
        const NodeDef& const_requantize_range_max_node = match.inputs[4].node;
        CHECK_EQ("Const", const_requantize_range_max_node.op());

        string quantized_matmul_op_name = quantized_matmul_node.op();
        // Set up the new fused version of the matmul op.
        NodeDef fused_matmul;
        fused_matmul.set_op(quantized_matmul_op_name + "AndRequantize");
        fused_matmul.set_name(match.node.name());
        std::string input_op = node_map[NodeNameFromInput(
          quantized_matmul_node.input(0))]->op();
        int n_input = quantized_matmul_node.input_size();
        for (int i = 0; i < n_input; i++)
          AddNodeInput(quantized_matmul_node.input(i), &fused_matmul);
        AddNodeInput(const_requantize_range_min_node.name(), &fused_matmul);
        AddNodeInput(const_requantize_range_max_node.name(), &fused_matmul);
        CopyNodeAttr(quantized_matmul_node, "T1", "T1", &fused_matmul);
        CopyNodeAttr(quantized_matmul_node, "T2", "T2", &fused_matmul);
        SetNodeAttr("Toutput", DT_QUINT8, &fused_matmul);
        SetNodeAttr("Tbias", DT_FLOAT, &fused_matmul);
        new_nodes->push_back(fused_matmul);
        new_nodes->push_back(const_requantize_range_min_node);
        new_nodes->push_back(const_requantize_range_max_node);
        return Status::OK();
      },
      {}, &replaced_graph_def));

  // Convert bias float -> int32 on replaced_graph_def
  std::vector<std::string> fused_requantized_bias_ops = {
        "QuantizedMatMulWithBias",
        "QuantizedMatMulWithBiasAndReluAndRequantize"
  };
  node_map.clear();
  MapNamesToNodes(replaced_graph_def, &node_map);
  for (auto& node_pair : node_map) {
    const NodeDef *node = node_pair.second;
    bool is_fused_requantized_matmul_op =
      std::find(fused_requantized_bias_ops.begin(),
                fused_requantized_bias_ops.end(),
      node->op()) != fused_requantized_bias_ops.end();
    if (is_fused_requantized_matmul_op) {
      // If the op is feed by Quantize op then we keep bias as float
      std::string input_op = node_map[NodeNameFromInput(
                    node->input(0))]->op();

      if (str_util::StartsWith(input_op, "QuantizedMatMul") &&
        str_util::EndsWith(input_op, "AndRequantize")) {
        NodeDef *bias_node = const_cast<NodeDef*>(node_map[NodeNameFromInput(
            node->input(2))]);

    const NodeDef *min_input_node = node_map[NodeNameFromInput(
            node_map[node->input(0)]->input(7))];
        const NodeDef *max_input_node = node_map[NodeNameFromInput(
            node_map[node->input(0)]->input(8))];

        const NodeDef *min_filter_node = node_map[NodeNameFromInput(
            node->input(5))];
        const NodeDef *max_filter_node = node_map[NodeNameFromInput(
            node->input(6))];
        const float min_input =
            GetNodeTensorAttr(*min_input_node, "value").flat<float>()(0);
        const float max_input =
            GetNodeTensorAttr(*max_input_node, "value").flat<float>()(0);
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

        float bias_scale = 255.0 * 127.0 /
            (std::max(std::abs(max_input), std::abs(min_input)) *
            std::max(std::abs(max_filter), std::abs(min_filter)));
        int64 nelems = float_tensor.NumElements();
        for (int64 n = 0; n < nelems; n++)
          p_bias_int32[n] = (int32_t) (p_bias_float[n] * bias_scale);

        bias_node->clear_attr();
        AttrValue attr_type;
        attr_type.set_type(int32_tensor.dtype());
        bias_node->mutable_attr()->insert({"dtype", attr_type});

        AttrValue attr_tensor;
        TensorProto* t = attr_tensor.mutable_tensor();
        int32_tensor.AsProtoTensorContent(t);
        bias_node->mutable_attr()->insert({"value", attr_tensor});
        SetNodeAttr("Tbias", DT_QINT32, const_cast<NodeDef*>(node));
    } else if (str_util::StartsWith(input_op, "QuantizeV2")) {
      NodeDef *bias_node = const_cast<NodeDef*>(node_map[NodeNameFromInput(
        node->input(2))]);
      NodeDef *weight_node = const_cast<NodeDef*>(node_map[NodeNameFromInput(
        node->input(1))]);
      const NodeDef *min_input_node = node_map[NodeNameFromInput(
        node_map[node->input(0)]->input(1))];
      const NodeDef *max_input_node = node_map[NodeNameFromInput(
        node_map[node->input(0)]->input(2))];

      const NodeDef *min_filter_node = node_map[NodeNameFromInput(
        node->input(5))];
      const NodeDef *max_filter_node = node_map[NodeNameFromInput(
        node->input(6))];

      const float min_input =
        GetNodeTensorAttr(*min_input_node, "value").flat<float>()(0);
      const float max_input =
        GetNodeTensorAttr(*max_input_node, "value").flat<float>()(0);
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

      TensorProto weight_int8_tensor_proto = weight_node->attr().at(
        "value").tensor();
      Tensor weight_int8_tensor;
      CHECK(weight_int8_tensor.FromProto(weight_int8_tensor_proto));
      CHECK_EQ(weight_int8_tensor.dtype(), DT_QINT8);
      qint8 *p_weight_int8 = weight_int8_tensor.flat<qint8>().data();

      int k = weight_int8_tensor.dim_size(0);
      int n = weight_int8_tensor.dim_size(1);

      float bias_scale = 255.0 * 127.0 /
        (std::max(std::abs(max_input), std::abs(min_input)) *
          std::max(std::abs(max_filter), std::abs(min_filter)));
      float QaAmin = 255 * min_input / (max_input - min_input);

      for (int j = 0; j < n; j++) {
        int x = 0;
        for (int i = 0; i < k; i++) {
          x += p_weight_int8[i * n + j];
        }

        p_bias_int32[j] = (int32_t)((p_bias_float[j] * bias_scale) +
          (x * QaAmin));
      }

      bias_node->clear_attr();
      AttrValue attr_type;
      attr_type.set_type(int32_tensor.dtype());
      bias_node->mutable_attr()->insert({ "dtype", attr_type });
      AttrValue attr_tensor;
      TensorProto* t = attr_tensor.mutable_tensor();
      int32_tensor.AsProtoTensorContent(t);
      bias_node->mutable_attr()->insert({ "value", attr_tensor });
      SetNodeAttr("Tbias", DT_QINT32, const_cast<NodeDef*>(node));
    } else {
        SetNodeAttr("Tbias", DT_FLOAT, const_cast<NodeDef*>(node));
      }
    }
  }

  *output_graph_def = replaced_graph_def;
  return Status::OK();
}

REGISTER_GRAPH_TRANSFORM("fuse_quantized_matmul_and_requantize",
                         FuseQuantizedMatMulAndRequantize);

}  // namespace graph_transforms
}  // namespace tensorflow
