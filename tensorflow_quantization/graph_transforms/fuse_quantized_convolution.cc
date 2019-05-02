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
#ifdef INTEL_MKL
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
#include "tensorflow/tools/graph_transforms/fold_constants_lib.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace graph_transforms {

Status FuseQuantizedConvolutionAndRequantize(
    const GraphDef& input_graph_def, const TransformFuncContext& context,
    GraphDef* output_graph_def) {
  std::map<string, const NodeDef*> node_map;
  MapNamesToNodes(input_graph_def, &node_map);
  bool is_perchannel = false;
  GraphDef replaced_graph_def;
  TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
      input_graph_def,  // clang-format off

      {"Requantize|RequantizePerChannel",
        {
          {"QuantizedConv2D|QuantizedConv2DWithBias|QuantizedConv2DWithRelu|"
            "QuantizedConv2DWithBiasAndRelu|QuantizedConv2DWithBiasSumAndRelu|"
            "QuantizedDepthwiseConv2DWithBiasAndRelu"},
          {"QuantizedConv2D|QuantizedConv2DWithBias|QuantizedConv2DWithRelu|"
           "QuantizedConv2DWithBiasAndRelu|QuantizedConv2DWithBiasSumAndRelu|"
           "QuantizedDepthwiseConv2DWithBiasAndRelu"},
          {"QuantizedConv2D|QuantizedConv2DWithBias|QuantizedConv2DWithRelu|"
           "QuantizedConv2DWithBiasAndRelu|QuantizedConv2DWithBiasSumAndRelu|"
           "QuantizedDepthwiseConv2DWithBiasAndRelu"},
          {"Const"},
          {"Const"}
        }
      },  // clang-format on */
      [&node_map, &is_perchannel](const NodeMatch& match,
         const std::set<string>& input_nodes,
         const std::set<string>& output_nodes,
         std::vector<NodeDef>* new_nodes) {
        // TODO(mdfaijul/sheng): Current implementation assumed all
        // requantization cases have bias. Index of inputs need to be updated
        // for non-bias cases.

        // Find all the nodes we expect in the subgraph.
        const NodeDef& requantize_node = match.node;
        const NodeDef& quantized_conv2D_node = match.inputs[0].node;
        const NodeDef& const_requantize_range_min_node = match.inputs[3].node;
        CHECK_EQ("Const", const_requantize_range_min_node.op());
        const NodeDef& const_requantize_range_max_node = match.inputs[4].node;
        CHECK_EQ("Const", const_requantize_range_max_node.op());

        is_perchannel = ("RequantizePerChannel" == requantize_node.op());

        string quantized_conv2D_op_name = quantized_conv2D_node.op();
        // Set up the new fused version of the convolution op.
        NodeDef fused_conv;
        fused_conv.set_op(quantized_conv2D_op_name + "AndRequantize");
        fused_conv.set_name(match.node.name());
        int n_input = quantized_conv2D_node.input_size();
        if (quantized_conv2D_op_name.compare(
                "QuantizedConv2DWithBiasSumAndRelu") == 0)
          n_input -= 1;  // -1 since summand is moved after frozen min-max

        string control_input;
        string current_input;
        for (int i=0; i < n_input; i++) {
          current_input = quantized_conv2D_node.input(i);
          if (current_input.length() > 0 && current_input[0] == '^') {
            control_input = current_input;
          } else {
            AddNodeInput(current_input, &fused_conv);
          }
        }
        AddNodeInput(const_requantize_range_min_node.name(), &fused_conv);
        AddNodeInput(const_requantize_range_max_node.name(), &fused_conv);

        // Add additional inputs to
        // QuantizedConv2DWithBiasSumAndReluAndRequantize
        if (quantized_conv2D_op_name.compare(
              "QuantizedConv2DWithBiasSumAndRelu") == 0) {
          const NodeDef *summand_node = node_map[quantized_conv2D_node.input(
            n_input)];
          NodeDef* new_summand_node = nullptr;
          NodeDef quantize_node;
          if (summand_node->op() != "Dequantize") {
            // Quantizing the summand.
            // Add some common constants we need for reshaping inputs.
            NodeDef reshape_dims;
            reshape_dims.set_op("Const");
            reshape_dims.set_name(summand_node->name() + "/reshape_dims");
            SetNodeAttr("dtype", DT_INT32, &reshape_dims);
            Tensor reshape_dims_tensor(DT_INT32, {1});
            reshape_dims_tensor.flat<int32>()(0) = -1;
            SetNodeTensorAttr<int32>(
              "value", reshape_dims_tensor, &reshape_dims);
            AddNodeInput("^" + summand_node->name(), &reshape_dims);

            NodeDef reduction_dims;
            reduction_dims.set_op("Const");
            reduction_dims.set_name(summand_node->name() + "/reduction_dims");
            SetNodeAttr("dtype", DT_INT32, &reduction_dims);
            Tensor reduction_dims_tensor(DT_INT32, {1});
            reduction_dims_tensor.flat<int32>()(0) = 0;
            SetNodeTensorAttr<int32>("value", reduction_dims_tensor,
                                    &reduction_dims);
            AddNodeInput("^" + summand_node->name(), &reduction_dims);

            NodeDef reshape_node;
            reshape_node.set_op("Reshape");
            reshape_node.set_name(summand_node->name() + "/reshape");
            SetNodeAttr("T", DT_FLOAT, &reshape_node);

            NodeDef min_node;
            min_node.set_op("Min");
            min_node.set_name(summand_node->name() + "/min");
            SetNodeAttr("T", DT_FLOAT, &min_node);
            SetNodeAttr("keep_dims", false, &min_node);
            AddNodeInput(reshape_node.name(), &min_node);
            AddNodeInput(reduction_dims.name(), &min_node);

            NodeDef max_node;
            max_node.set_op("Max");
            max_node.set_name(summand_node->name() + "/max");
            SetNodeAttr("T", DT_FLOAT, &max_node);
            SetNodeAttr("keep_dims", false, &max_node);
            AddNodeInput(reshape_node.name(), &max_node);
            AddNodeInput(reduction_dims.name(), &max_node);

            // NodeDef quantize_node;
            quantize_node.set_op("QuantizeV2");
            quantize_node.set_name(summand_node->name() + "/quantize");
            // Decide data type of quantize op
            std::vector<string> relu_ops = {
                "Relu",
                "Relu6"
                };
            bool is_relu = std::find(relu_ops.begin(), relu_ops.end(),
                          summand_node->op()) != relu_ops.end();
            if (is_relu)
              SetNodeAttr("T", DT_QUINT8, &quantize_node);
            else
              SetNodeAttr("T", DT_QINT8, &quantize_node);
            SetNodeAttr("mode", "SCALED", &quantize_node);

            AddNodeInput(summand_node->name(), &reshape_node);
            AddNodeInput(reshape_dims.name(), &reshape_node);

            AddNodeInput(summand_node->name(), &quantize_node);
            AddNodeInput(min_node.name(), &quantize_node);
            AddNodeInput(max_node.name(), &quantize_node);

            new_nodes->push_back(reshape_dims);
            new_nodes->push_back(reduction_dims);
            new_nodes->push_back(reshape_node);
            new_nodes->push_back(min_node);
            new_nodes->push_back(max_node);
            new_nodes->push_back(quantize_node);
            // Set the new summand node for fused_conv
            new_summand_node = &quantize_node;
          } else {
            // If summand node is Dequantize then either QuantizeV2 or
            // Requantize{PerChannel} is feeding Dequantize op
            new_summand_node = const_cast<NodeDef*>(node_map[
                  summand_node->input(0)]);
          }
          string summand(new_summand_node->name());
          string min_summand(new_summand_node->name() + ":1");
          string max_summand(new_summand_node->name() + ":2");
          AddNodeInput(summand, &fused_conv);
          AddNodeInput(min_summand, &fused_conv);
          AddNodeInput(max_summand, &fused_conv);

          DataType summand_type;
          // New summand node should be QuantizeV2 or
          // Requantize{PerChannel}
          if (new_summand_node->op() == "QuantizeV2") {
            TF_RETURN_IF_ERROR(GetNodeAttr(*new_summand_node,
                                           "T", &summand_type));
          } else if (new_summand_node->op() == "Requantize" ||
                     new_summand_node->op() == "RequantizePerChannel") {
            TF_RETURN_IF_ERROR(GetNodeAttr(*new_summand_node,
                                           "out_type", &summand_type));
          } else {
            return Status(error::Code::FAILED_PRECONDITION,
                               "Fusion is not supported, a fix is required.");
          }
          SetNodeAttr("Tsummand", summand_type, &fused_conv);
          // Decide whether signed version of
          // QuantizedConv2DWithBiasSumAndReluAndRequantize or not
          if (summand_type == DT_QINT8)
            fused_conv.set_op(
                "QuantizedConv2DWithBiasSignedSumAndReluAndRequantize");
        }

        // Add control input to the very end of the input list
        // of the newly fused op
        if (control_input.length() > 0)
          AddNodeInput(control_input, &fused_conv);

        CopyNodeAttr(quantized_conv2D_node, "Tinput", "Tinput",   &fused_conv);
        CopyNodeAttr(quantized_conv2D_node, "Tfilter", "Tfilter", &fused_conv);
        CopyNodeAttr(quantized_conv2D_node, "strides", "strides", &fused_conv);
        CopyNodeAttr(quantized_conv2D_node, "padding", "padding", &fused_conv);

        std::vector<std::string> fused_quantized_bias_ops = {
          "QuantizedConv2DWithBias",
          "QuantizedConv2DWithBiasAndRelu",
          "QuantizedDepthwiseConv2DWithBias",
          "QuantizedDepthwiseConv2DWithBiasAndRelu",
          "QuantizedConv2DWithBiasSumAndRelu",
          "QuantizedConv2DWithBiasSignedSumAndRelu"
        };

        if (std::find(fused_quantized_bias_ops.begin(),
            fused_quantized_bias_ops.end(),
            quantized_conv2D_node.op()) != fused_quantized_bias_ops.end()) {
          SetNodeAttr("Tbias", DT_FLOAT, &fused_conv);
        }

        if (HasNodeAttr(quantized_conv2D_node, "padding_list"))
          CopyNodeAttr(quantized_conv2D_node, "padding_list",
                       "padding_list",     &fused_conv);
        // Copy dilation attribute if exsit in the orginal node
        if (HasNodeAttr(quantized_conv2D_node, "dilations"))
          CopyNodeAttr(quantized_conv2D_node, "dilations",
                       "dilations", &fused_conv);
        if (quantized_conv2D_op_name.compare("QuantizedConv2D") == 0 ||
           quantized_conv2D_op_name.compare("QuantizedConv2DWithBias") == 0)
          SetNodeAttr("out_type", DT_QINT8, &fused_conv);
        else
          SetNodeAttr("out_type", DT_QUINT8, &fused_conv);
        new_nodes->push_back(fused_conv);
        new_nodes->push_back(const_requantize_range_min_node);
        new_nodes->push_back(const_requantize_range_max_node);

        return Status::OK();
      },
      {}, &replaced_graph_def));

  if (!is_perchannel) {
    // Convert bias float -> int32 on replaced_graph_def
    std::vector<std::string> fused_requantized_bias_ops = {
        "QuantizedConv2DWithBiasAndRequantize",
        "QuantizedConv2DWithBiasAndReluAndRequantize",
        "QuantizedConv2DWithBiasSumAndReluAndRequantize",
        "QuantizedConv2DWithBiasSignedSumAndReluAndRequantize"
    };

    node_map.clear();
    MapNamesToNodes(replaced_graph_def, &node_map);
    for (auto& node_pair : node_map) {
      const NodeDef *node = node_pair.second;
      if (str_util::StartsWith(node->op(), "Dequantize")) {
        // dequant node should accept DT_QINT8 if the input node is
        // "QuantizedConv2DAndRequantize" and
        //  "QuantizedConv2DWithBiasAndRequantize"
        std::string input_node_op =
            node_map[NodeNameFromInput(node->input(0))]->op();
        if (str_util::StartsWith(input_node_op,
               "QuantizedConv2DAndRequantize") ||
            str_util::StartsWith(input_node_op,
               "QuantizedConv2DWithBiasAndRequantize")) {
          SetNodeAttr("T", DT_QINT8, const_cast<NodeDef*>(node));
          SetNodeAttr("mode", "SCALED", const_cast<NodeDef*>(node));
        }
      continue;
    }

    bool is_fused_requantized_conv_op =
      std::find(fused_requantized_bias_ops.begin(),
                fused_requantized_bias_ops.end(), node->op())
          != fused_requantized_bias_ops.end();
      if (is_fused_requantized_conv_op) {
        // If the op is feed by Quantize op then we keep bias as float
        std::string input_op = node_map[NodeNameFromInput(
                    node->input(0))]->op();
        if (str_util::StartsWith(input_op, "QuantizedConv2D") &&
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

          TensorProto float_tensor_proto =
              bias_node->attr().at("value").tensor();
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
        } else {
          SetNodeAttr("Tbias", DT_FLOAT, const_cast<NodeDef*>(node));
        }
      }
    }
  }
  *output_graph_def = replaced_graph_def;
  return Status::OK();
}

REGISTER_GRAPH_TRANSFORM("fuse_quantized_conv_and_requantize",
                         FuseQuantizedConvolutionAndRequantize);

}  // namespace graph_transforms
}  // namespace tensorflow
#endif  // INTEL_MKL
