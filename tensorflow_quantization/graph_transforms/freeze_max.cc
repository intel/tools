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

#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace graph_transforms {

struct MaxRecord {
  string name;
  float max;
};

// Try to parse a log file containing loosely-structured lines, some of which
// are the min/max logs we want.
Status ExtractMaxRecords(const string& log_file_name,
                         std::vector<MaxRecord>* records) {
  string file_data;
  TF_RETURN_IF_ERROR(
      ReadFileToString(Env::Default(), log_file_name, &file_data));
  const string print_suffix("__print__");
  const string requant_prefix("__max:");
  std::vector<string> file_lines = str_util::Split(file_data, '\n');
  for (const string& file_line : file_lines) {
    // We expect to find a line with components separated by semicolons, so to
    // start make sure that the basic structure is in place/
    if (!str_util::StrContains(file_line,
                               print_suffix + ";" + requant_prefix)) {
      continue;
    }

    std::vector<string> line_parts = str_util::Split(file_line, ';');
    if (line_parts.size() < 2) {
      continue;
    }

    // Now we want to figure out which components have the name and min max
    // values by scanning for the prefix we expect.
    bool min_max_found = false;
    int min_max_index;
    for (int i = 1; i < line_parts.size(); ++i) {
      if (str_util::StartsWith(line_parts[i], requant_prefix)) {
        min_max_found = true;
        min_max_index = i;
      }
    }
    if (!min_max_found) {
      continue;
    }
    // Finally we need to break out the values from the strings, and parse them
    // into a form we can use.
    string min_max_string = line_parts[min_max_index];
    std::vector<string> min_max_parts = str_util::Split(min_max_string, '[');
    if ((min_max_parts.size() != 2) || (min_max_parts[0] != requant_prefix)) {
      continue;
    }
    string max_string = min_max_parts[1];
    std::vector<string> max_string_parts = str_util::Split(max_string, ']');
    if (max_string_parts.size() != 2) {
      continue;
    }
    string max_number_string = max_string_parts[0];
    float max;
    if (!strings::safe_strtof(max_number_string.c_str(), &max)) {
      continue;
    }
    StringPiece name_string = line_parts[min_max_index - 1];
    if (!str_util::EndsWith(name_string, print_suffix)) {
      continue;
    }
    string name(
        name_string.substr(0, name_string.size() - print_suffix.size()));
    records->push_back({name, max});
  }
  return Status::OK();
}

// Uses the observed min/max values for requantization captured in a log file to
// replace costly RequantizationRange ops with simple Consts.
Status FreezeMax(const GraphDef& input_graph_def,
                 const TransformFuncContext& context,
                 GraphDef* output_graph_def) {
  string min_max_log_file;
  TF_RETURN_IF_ERROR(
      context.GetOneStringParameter("min_max_log_file", "", &min_max_log_file));
  if (min_max_log_file.empty()) {
    return errors::InvalidArgument(
        "You must pass a file name to min_max_log_file");
  }
  float min_percentile;
  TF_RETURN_IF_ERROR(
      context.GetOneFloatParameter("min_percentile", 5.0f, &min_percentile));
  float max_percentile;
  TF_RETURN_IF_ERROR(
      context.GetOneFloatParameter("max_percentile", 5.0f, &max_percentile));

  std::vector<MaxRecord> records;
  TF_RETURN_IF_ERROR(ExtractMaxRecords(min_max_log_file, &records));
  if (records.empty()) {
    return errors::InvalidArgument(
        "No min/max range logs were found in the log file");
  }
  std::map<string, const NodeDef*> node_map;
  MapNamesToNodes(input_graph_def, &node_map);
  bool any_missing_nodes = false;
  std::map<string, std::vector<MaxRecord>> records_by_node;
  for (const MaxRecord& record : records) {
    records_by_node[record.name].push_back(record);
    if (!node_map.count(record.name)) {
      any_missing_nodes = true;
      LOG(WARNING) << "Node from log not found in graph: " << record.name;
    }
  }
  if (any_missing_nodes) {
    return errors::InvalidArgument(
        "Nodes were found in the log file that aren't present in the graph");
  }

  // Now find out the largest and smallest min/max values for the node.
  std::map<string, float> range_for_nodes;
  for (const auto& record_info : records_by_node) {
    const string& name = record_info.first;
    const std::vector<MaxRecord> records = record_info.second;
    std::vector<float> mins;
    std::vector<float> maxs;
    for (const MaxRecord& record : records) {
      maxs.push_back(record.max);
    }
    std::sort(maxs.begin(), maxs.end());
    int max_index =
        std::round(maxs.size() * (1.0f - (max_percentile / 100.0f)));
    if (max_index > (maxs.size() - 1)) {
      max_index = maxs.size() - 1;
    }
    const float max = maxs[max_index];
    range_for_nodes[name] = {max};
  }
  std::map<string, string> inputs_to_rename;
  GraphDef frozen_graph_def;
  for (const NodeDef& node : input_graph_def.node()) {
    if (range_for_nodes.count(node.name())) {
      if (node.op() != "Max") {
        return errors::InvalidArgument("Node is expected to be a Max op: ",
                                       node.name(), " , but is: ", node.op());
      }
      const float max_value = range_for_nodes.at(node.name());
      NodeDef* max_node = frozen_graph_def.mutable_node()->Add();
      max_node->set_op("Const");
      max_node->set_name(node.name() + "/frozen_max_only");
      SetNodeAttr("dtype", DT_FLOAT, max_node);
      Tensor max_tensor(DT_FLOAT, {});
      max_tensor.flat<float>()(0) = max_value;
      SetNodeTensorAttr<float>("value", max_tensor, max_node);
      inputs_to_rename[node.name()] = max_node->name() + ":0";
    } else {
      NodeDef* new_node = frozen_graph_def.mutable_node()->Add();
      *new_node = node;
    }
  }
  return RenameNodeInputs(frozen_graph_def, inputs_to_rename,
                          std::unordered_set<string>(), output_graph_def);
}

REGISTER_GRAPH_TRANSFORM("freeze_max", FreezeMax);

}  // namespace graph_transforms
}  // namespace tensorflow
