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

 SPDX-License-Identifier: EPL-2.0
==============================================================================*/

#include <algorithm>

#include "tensorflow/tools/graph_transforms/fold_constants_lib.h"
#include "tensorflow/core/common_runtime/constant_folding.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/subgraph.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace graph_transforms {

Status MklFusePadAndConv(const GraphDef& input_graph_def,
                      const TransformFuncContext& context,
                      GraphDef *output_graph_def) {
  GraphDef replaced_graph_def;
  TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
      input_graph_def,  // clang-format off

      {"QuantizeV2",
          {
              {"Relu",
                  {
                      {"BiasAdd",
                          {
                              {"Conv2D",
                                  {
                                      {"Pad",
                                          {
                                              {"Dequantize"},
                                              {"Const"}
                                          }
                                      },
                                      {"Const"}
                                  }
                              },
                              {"Const"}
                          }
                      }
                  }
              },
              {"Min",
                  {
                      {"Reshape",
                          {
                            {"Relu"},
                            {"Const"}
                          }
                      },
                      {"Const"}
                  }
              },
              {"Max"}
          }

      }, // clang-format on */
      [](const NodeMatch& match, const std::set<string>& input_nodes,
         const std::set<string>& output_nodes,
         std::vector<NodeDef>* new_nodes) {

        // Find all the nodes we expect in the subgraph.
        const NodeDef& quantize_v2_node = match.node;
        CHECK_EQ("QuantizeV2", quantize_v2_node.op());
        const NodeDef& relu_node = match.inputs[0].node;
        CHECK_EQ("Relu", relu_node.op());
        const NodeDef& bias_add_node = match.inputs[0].inputs[0].node;
        CHECK_EQ("BiasAdd", bias_add_node.op());
        const NodeDef& conv2d_node = match.inputs[0].inputs[0].inputs[0].node;
        CHECK_EQ("Conv2D", conv2d_node.op());
        const NodeDef& const_bias_node = match.inputs[0].inputs[0].inputs[1].node;
        CHECK_EQ("Const", const_bias_node.op());
        const NodeDef& pad_node = match.inputs[0].inputs[0].inputs[0].inputs[0].node;
        CHECK_EQ("Pad", pad_node.op());
        const NodeDef& const_filter_node = match.inputs[0].inputs[0].inputs[0].inputs[1].node;
        CHECK_EQ("Const", const_filter_node.op());
        const NodeDef& dequantize_node = match.inputs[0].inputs[0].inputs[0].inputs[0].inputs[0].node;
        CHECK_EQ("Dequantize", dequantize_node.op());
        const NodeDef& const_paddings_node = match.inputs[0].inputs[0].inputs[0].inputs[0].inputs[1].node;
        CHECK_EQ("Const", const_paddings_node.op());

        // Read the value on convolution's padding attribute
        string padding_value;
        auto attr = conv2d_node.attr().at("padding");
        attr.SerializeToString(&padding_value);

        // This pad fusion transform only supports VALID padding
        // as input to convolution
        if (str_util::StrContains(padding_value, "VALID")) {
          // Use the original bias node
          new_nodes->push_back(const_bias_node);

          // Create new Requantize and RequantizationRange nodes
          NodeDef requantize;
          requantize.set_op("Requantize");
          requantize.set_name(match.node.name());
          SetNodeAttr("Tinput", DT_QINT32, &requantize);
          SetNodeAttr("out_type", DT_QUINT8, &requantize);

          NodeDef req_range;
          req_range.set_op("RequantizationRange");
          req_range.set_name(match.node.name() + "/pad_fusion_requantization_range");
          SetNodeAttr("Tinput", DT_QINT32, &req_range);


          // Create the new fused Convolution node
          NodeDef fused_pad_and_conv;
          fused_pad_and_conv.set_op("QuantizedConv2DWithBiasAndRelu");
          fused_pad_and_conv.set_name(match.node.name() + "/fused_pad_and_conv");

          // Connect nodes' inputs
          AddNodeInput(fused_pad_and_conv.name(), &requantize);
          AddNodeInput(fused_pad_and_conv.name() + ":1", &requantize);
          AddNodeInput(fused_pad_and_conv.name() + ":2", &requantize);
          AddNodeInput(req_range.name(), &requantize);
          AddNodeInput(req_range.name() + ":1", &requantize);

          AddNodeInput(fused_pad_and_conv.name(), &req_range);
          AddNodeInput(fused_pad_and_conv.name() + ":1", &req_range);
          AddNodeInput(fused_pad_and_conv.name() + ":2", &req_range);

          SetNodeAttr("Tinput", DT_QUINT8, &fused_pad_and_conv);
          SetNodeAttr("Tfilter", DT_QINT8, &fused_pad_and_conv);
          SetNodeAttr("out_type", DT_QINT32, &fused_pad_and_conv);

          CopyNodeAttr(conv2d_node, "strides", "strides", &fused_pad_and_conv);
          CopyNodeAttr(conv2d_node, "padding", "padding", &fused_pad_and_conv);

          if (HasNodeAttr(conv2d_node, "dilations"))
            CopyNodeAttr(conv2d_node, "dilations", "dilations", &fused_pad_and_conv);

          // Quantize the convolution filter
          TensorProto filter_tensor_proto = const_filter_node.attr().at("value").tensor();
          Tensor filter_tensor;
          CHECK(filter_tensor.FromProto(filter_tensor_proto));
          float* filter_buf = filter_tensor.flat<float>().data();


          Tensor output_filter(DT_QINT8, filter_tensor.shape());
          qint8* output_filter_buf = output_filter.flat<qint8>().data();

          Eigen::Tensor<float, 0, Eigen::RowMajor> min = filter_tensor.flat_inner_dims<float>().minimum();
          Eigen::Tensor<float, 0, Eigen::RowMajor> max = filter_tensor.flat_inner_dims<float>().maximum();
          float filter_min = min();
          float filter_max = max();
          float min_range, max_range;
          min_range = std::min(0.0f, filter_min);
          const float epsilon = std::max(1.0f, std::max(fabsf(filter_min),
                                                  fabsf(filter_max))) /
                          100.0f;
          max_range = std::max(filter_max, min_range + epsilon);
          max_range = std::max(0.0f, max_range);
          float scale = std::max(std::abs(min_range), std::abs(max_range));

          for (int i=0; i < filter_tensor.NumElements(); i++)
          {
            output_filter_buf[i] = static_cast<qint8>(round(127.0f * filter_buf[i] / scale));
          }

          // Create and set up the quantized filter node
          NodeDef quantized_filter_node;
          quantized_filter_node.set_op("Const");
          quantized_filter_node.set_name(match.node.name() + "/pad_fusion_quantized_filter");
          quantized_filter_node.clear_attr();
          AttrValue attr_type;
          attr_type.set_type(output_filter.dtype());
          quantized_filter_node.mutable_attr()->insert({"dtype", attr_type});

          AttrValue attr_tensor;
          TensorProto* t = attr_tensor.mutable_tensor();
          output_filter.AsProtoTensorContent(t);
          quantized_filter_node.mutable_attr()->insert({"value", attr_tensor});

          // Create and set up a filter_min node
          NodeDef quantized_filter_min;
          quantized_filter_min.set_op("Const");
          quantized_filter_min.set_name(match.node.name() + "/pad_fusion_filter_min");
          AttrValue attr_type_filter_min;
          attr_type_filter_min.set_type(DT_FLOAT);
          quantized_filter_min.mutable_attr()->insert({"dtype", attr_type_filter_min});

          TensorShape scalar_tensor_shape;
          int32 dims_array = 0;
          TensorShapeUtils::MakeShape(&dims_array, 0, &scalar_tensor_shape);
          Tensor filter_min_tensor(DT_FLOAT, scalar_tensor_shape);
          float* filter_min_buf = filter_min_tensor.flat<float>().data();
          filter_min_buf[0] = scale * -1.0f;
          AttrValue attr_tensor_filter_min;
          TensorProto* t_filter_min = attr_tensor_filter_min.mutable_tensor();
          filter_min_tensor.AsProtoTensorContent(t_filter_min);
          quantized_filter_min.mutable_attr()->insert({"value", attr_tensor_filter_min});

          // Create and set up a filter_max node
          NodeDef quantized_filter_max;
          quantized_filter_max.set_op("Const");
          quantized_filter_max.set_name(match.node.name() + "/pad_fusion_filter_max");
          AttrValue attr_type_filter_max;
          attr_type_filter_max.set_type(DT_FLOAT);
          quantized_filter_max.mutable_attr()->insert({"dtype", attr_type_filter_max});

          Tensor filter_max_tensor(DT_FLOAT, scalar_tensor_shape);
          float* filter_max_buf = filter_max_tensor.flat<float>().data();
          filter_max_buf[0] = scale;
          AttrValue attr_tensor_filter_max;
          TensorProto* t_filter_max = attr_tensor_filter_max.mutable_tensor();
          filter_max_tensor.AsProtoTensorContent(t_filter_max);
          quantized_filter_max.mutable_attr()->insert({"value", attr_tensor_filter_max});

          // Add the quantized filter, filter_min and filter_max nodes to the replacement
          new_nodes->push_back(quantized_filter_node);
          new_nodes->push_back(quantized_filter_min);
          new_nodes->push_back(quantized_filter_max);
          AddNodeInput(dequantize_node.input(0), &fused_pad_and_conv);
          AddNodeInput(quantized_filter_node.name(), &fused_pad_and_conv);
          AddNodeInput(const_bias_node.name(), &fused_pad_and_conv);
          AddNodeInput(dequantize_node.input(1), &fused_pad_and_conv);
          AddNodeInput(dequantize_node.input(2), &fused_pad_and_conv);
          AddNodeInput(quantized_filter_min.name(), &fused_pad_and_conv);
          AddNodeInput(quantized_filter_max.name(), &fused_pad_and_conv);

          // Pass the paddings (Pad op's input) to the convolution fusion as an attribute
          TensorProto paddings_tensor_proto = const_paddings_node.attr().at("value").tensor();
          Tensor paddings_tensor;
          CHECK(paddings_tensor.FromProto(paddings_tensor_proto));
          int *paddings_data = paddings_tensor.flat<int>().data();

          std::vector<int> pad_list;
          for (int i=0; i < paddings_tensor.NumElements(); i++) {
            pad_list.push_back(paddings_data[i]);
          }

          SetNodeAttr("padding_list", pad_list, &fused_pad_and_conv);

          // Add the 3 new nodes to the replacement sub-graph
          new_nodes->push_back(fused_pad_and_conv);
          new_nodes->push_back(requantize);
          new_nodes->push_back(req_range);
        }
        else {
          CopyOriginalMatch(match, new_nodes);
          LOG(ERROR) << "Pad fusion only supports VALID padding. Pattern Skipped!.";
        }
        return Status::OK();
      },
      {}, &replaced_graph_def));
  *output_graph_def = replaced_graph_def;
  return Status::OK();
}

REGISTER_GRAPH_TRANSFORM("mkl_fuse_pad_and_conv",
                         MklFusePadAndConv);

}  // namespace graph_transforms
}  // namespace tensorflow
