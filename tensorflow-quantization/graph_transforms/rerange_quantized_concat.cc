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

Status RerangeQuantizedConcat(const     GraphDef& input_graph_def,
                              const     TransformFuncContext& context,
                              GraphDef *output_graph_def) {
  *output_graph_def = input_graph_def;
  std::map<string, const NodeDef*> node_map;
  MapNamesToNodes(*output_graph_def, &node_map);

  std::map<string, int> offset_map;
  offset_map["QuantizedConv2DAndRequantize"] = 6;
  offset_map["QuantizedConv2DAndReluAndRequantize"] = 6;
  offset_map["QuantizedConv2DWithBiasAndRequantize"] = 7;
  offset_map["QuantizedConv2DWithBiasAndReluAndRequantize"] = 7;
  offset_map["QuantizedConv2DWithBiasSumAndReluAndRequantize"] = 7;
  offset_map["QuantizedConv2DWithBiasSignedSumAndReluAndRequantize"] = 7;

  for (auto& node_pair : node_map) {
    const NodeDef *node = node_pair.second;
    if (node->op().compare("QuantizedConcatV2") != 0)
      continue;

    const NodeDef& quantized_concat_node = *node;

    int n_input;
    GetNodeAttr(quantized_concat_node, "N", &n_input);

    // check if all inputs are from fused quantized conv2d and requantize
    // if they are not, do not do the rerange.
    bool all_fused_conv2d_and_requantize = true;
    for (int i = 0; i < n_input; i++) {
      const NodeDef *conv = node_map[quantized_concat_node.input(i)];
      if (offset_map.find(conv->op()) == offset_map.end()) {
        all_fused_conv2d_and_requantize = false;
      }
    }

    if (!all_fused_conv2d_and_requantize) {
      printf("Can not re-range the inputs for node %s\n", node->name().c_str());
      continue;
    }

    // find the combined range from all input tensors
    float combined_min = 0.0f, combined_max = 0.0f;
    for (int i = 0; i < n_input; i++) {
      const NodeDef *conv = node_map[quantized_concat_node.input(i)];
      int min_offset = offset_map[conv->op()];
      const NodeDef *min_freezed_output_node =
          node_map[conv->input(min_offset)];
      const NodeDef *max_freezed_output_node =
          node_map[conv->input(min_offset+1)];

      const float min_freezed_output =
        GetNodeTensorAttr(*min_freezed_output_node, "value").flat<float>()(0);
      const float max_freezed_output =
        GetNodeTensorAttr(*max_freezed_output_node, "value").flat<float>()(0);
      if (i == 0) {
        combined_min = min_freezed_output;
        combined_max = max_freezed_output;
      } else {
        combined_min = min_freezed_output < combined_min
                       ? min_freezed_output : combined_min;
        combined_max = max_freezed_output < combined_max
                       ? max_freezed_output : combined_max;
      }
    }

    // set the combined range to quantized convolution node
    Tensor min_tensor(DT_FLOAT, {});
    min_tensor.flat<float>()(0) = combined_min;
    Tensor max_tensor(DT_FLOAT, {});
    max_tensor.flat<float>()(0) = combined_max;

    for (int i = 0; i < n_input; i++) {
      const NodeDef *conv = node_map[quantized_concat_node.input(i)];
      int min_offset = offset_map[conv->op()];
      NodeDef *min_freezed_output_node =
          (NodeDef*) node_map[conv->input(min_offset)];
      NodeDef *max_freezed_output_node =
          (NodeDef*) node_map[conv->input(min_offset+1)];
      SetNodeTensorAttr<float>("value", min_tensor, min_freezed_output_node);
      SetNodeTensorAttr<float>("value", max_tensor, max_freezed_output_node);
    }
  }

  return Status::OK();
}

REGISTER_GRAPH_TRANSFORM("rerange_quantized_concat",
                         RerangeQuantizedConcat);

}  // namespace graph_transforms
}  // namespace tensorflow
