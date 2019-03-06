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
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/tools/graph_transforms/fold_constants_lib.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace graph_transforms {

// Convolution is:
// ---------------
// A quick recap of what a convolution layer calculates: if x is the
// pixels in the input image and w is the weights for the layer, then
// the convolution basically computes the following for each output pixel:
// out[j] = x[i]*w[0] + x[i+1]*w[1] + x[i+2]*w[2] + ... + x[i+k]*w[k] + b
// So x[i] is the input to Conv2D.
//    w[i] is the Const to Conv2D.
//
// Mul After Convolution where out[j] is output of Conv2D
//           out[j] * mul
// where 'mul' is the scalar const of Mul Op.
//
// Since there is no quantized Mul op, to avoid unnecessary
// quantization and dequantization, Mul scalar const be
// folded into Conv2D and BiasAdd in the following way
//           w * mul
//           b * mul
// With this folding, Mul op can be eliminated from the result fp32 graph.
//
// This transform should be done from one fp32 graph to another fp32 graph,
// before actual graph quantization.
// For instance, this graph transform can be applied to the
// inception_resnets_v2 slim model.

Status FoldConvolutionAndBiasWithMul(
    const GraphDef& input_graph_def, const TransformFuncContext& context,
    GraphDef* output_graph_def) {
  GraphDef replaced_graph_def;
  TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
      input_graph_def,  // clang-format off

      {"Add",
        {
          {"*"},                    // ConcatV2, Relu, or other
          {"Mul",
            {
              {"BiasAdd",           // Realdiv node
                {
                  {"Conv2D",        // conv2d node
                    {
                      {"*"},        // src
                      {"Const"}     // const weights
                    }
                  },
                  {"Const"}         // const bias
                }
              },
              {"Const"}             // const mul
            }
          }
        }
      },  // clang-format on */
      [](const NodeMatch& match, const std::set<string>& input_nodes,
         const std::set<string>& output_nodes,
         std::vector<NodeDef>* new_nodes) {
        // Find all the nodes we expect in the subgraph.
        const NodeDef& add_node     = match.node;
        const NodeDef& mul_c_node   = match.inputs[1].inputs[1].node;
        const NodeDef& biasadd_node = match.inputs[1].inputs[0].node;
        const NodeDef& bias_node    = match.inputs[1].inputs[0].inputs[1].node;
        const NodeDef& conv_node    = match.inputs[1].inputs[0].inputs[0].node;
        const NodeDef& weights_node =
            match.inputs[1].inputs[0].inputs[0].inputs[1].node;
        const NodeDef& add_input_node  = match.inputs[0].node;
        const NodeDef& conv_input_node =
            match.inputs[1].inputs[0].inputs[0].inputs[0].node;

        // Verfiy all the const nodes
        CHECK_EQ("Const", mul_c_node.op());
        CHECK_EQ("Const", bias_node.op());
        CHECK_EQ("Const", weights_node.op());

        // Get the Tensor values of all the constant nodes
        Tensor mul_c_tensor   = GetNodeTensorAttr(mul_c_node, "value");
        Tensor bias_tensor    = GetNodeTensorAttr(bias_node, "value");
        Tensor weights_tensor = GetNodeTensorAttr(weights_node, "value");

        // Locate data buffer
        float* mul_c_data = mul_c_tensor.flat<float>().data();
        float* bias_data = bias_tensor.flat<float>().data();
        float* weights_data = weights_tensor.flat<float>().data();

        // Mul multiplier
        float multipler = mul_c_data[0];

        // Apply multiplier on bias data
        Tensor new_bias_tensor(DT_FLOAT, bias_tensor.shape());
        float* new_bias_data = new_bias_tensor.flat<float>().data();
        for (int k=0; k < new_bias_tensor.NumElements(); k++)
           new_bias_data[k] = bias_data[k] * multipler;

        // Apply multiplier on weights data
        Tensor new_weights_tensor(DT_FLOAT, weights_tensor.shape());
        float* new_weights_data = new_weights_tensor.flat<float>().data();
        for (int k=0; k < new_weights_tensor.NumElements(); k++)
           new_weights_data[k] = weights_data[k] * multipler;

        // Create the new weights const node.
        NodeDef new_weights_node;
        new_weights_node.set_op("Const");
        new_weights_node.set_name(weights_node.name());
        SetNodeAttr("dtype", DT_FLOAT, &new_weights_node);
        SetNodeTensorAttr<float>("value", new_weights_tensor,
            &new_weights_node);

        // Create the new bias const node.
        NodeDef new_bias_node;
        new_bias_node.set_op("Const");
        new_bias_node.set_name(bias_node.name());
        SetNodeAttr("dtype", DT_FLOAT, &new_bias_node);
        SetNodeTensorAttr<float>("value", new_bias_tensor, &new_bias_node);

        NodeDef new_add_node;
        new_add_node.set_op("Add");
        new_add_node.set_name(add_node.name());
        CopyNodeAttr(add_node, "T", "T", &new_add_node);
        AddNodeInput(add_input_node.name(), &new_add_node);
        AddNodeInput(biasadd_node.name(), &new_add_node);

        new_nodes->push_back(conv_input_node);
        new_nodes->push_back(new_weights_node);
        new_nodes->push_back(conv_node);
        new_nodes->push_back(new_bias_node);
        new_nodes->push_back(biasadd_node);
        new_nodes->push_back(add_input_node);
        new_nodes->push_back(new_add_node);

        return Status::OK();
      },
      {}, &replaced_graph_def));


  *output_graph_def = replaced_graph_def;
  return Status::OK();
}
REGISTER_GRAPH_TRANSFORM("fold_convolutionwithbias_mul",
                         FoldConvolutionAndBiasWithMul);

}  // namespace graph_transforms
}  // namespace tensorflow
#endif  // INTEL_MKL
