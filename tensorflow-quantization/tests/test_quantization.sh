#
# Copyright (c) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: EPL-2.0
#

#!/usr/bin/env bash

echo 'Running with parameters:'
echo "    WORKSPACE: ${WORKSPACE}"
echo "    TF_WORKSPACE: ${TF_WORKSPACE}"
echo "    TEST_WORKSPACE: ${TEST_WORKSPACE}"
echo "    Mounted Volumes:"
echo "        ${PRE_TRAINED_MODEL_DIR} mounted on: ${MOUNT_OUTPUT}"

# output directory for tests
OUTPUT=${MOUNT_OUTPUT}

function test_ouput_graph(){
    test -f ${OUTPUT_GRAPH}
}

# model quantization steps
function run_quantize_model_test(){

    # Get the dynamic range int8 graph
    echo "Generate the dynamic range int8 graph for ${model} model..."
    cd ${TF_WORKSPACE}
    python tensorflow/tools/quantization/quantize_graph.py \
    --input=${FP32_MODEL} \
    --output=${OUTPUT}/${model}_int8_dynamic_range_graph.pb \
    --output_node_names=${OUTPUT_NODES} \
    --mode=eightbit \
    --intel_cpu_eightbitize=True \
    --model_name=${MODEL_NAME} \
    ${EXTRA_ARG}

    echo "${model}_int8_dynamic_range_graph.pb is successfully created."
    OUTPUT_GRAPH=${OUTPUT}/${model}_int8_dynamic_range_graph.pb test_ouput_graph

    if [ ${model}=="rfcn" ]; then
        # Apply Pad Fusion optimization:
        bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
        --in_graph=${OUTPUT}/${model}_int8_dynamic_range_graph.pb \
        --out_graph=${OUTPUT}/${model}_int8_dynamic_range_graph.pb \
        --outputs=${OUTPUT_NODES} \
        --transforms='mkl_fuse_pad_and_conv'

        OUTPUT_GRAPH=${OUTPUT}/${model}_int8_dynamic_range_graph.pb test_ouput_graph
    fi

    # Generate graph with logging
    echo "Generate the graph with logging for ${model} model..."
    bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
    --in_graph=/${OUTPUT}/${model}_int8_dynamic_range_graph.pb \
    --out_graph=${OUTPUT}/${model}_int8_logged_graph.pb \
    --transforms="${TRANSFORMS1}"

    echo "${model}_int8_logged_graph.pb is successfully created."
    OUTPUT_GRAPH=${OUTPUT}/${model}_int8_logged_graph.pb test_ouput_graph

    # Convert the dynamic range int8 graph to freezed range graph
    echo "Freeze the dynamic range graph using the min max constants from ${model}_min_max_log.txt..."
    bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
    --in_graph=/${OUTPUT}/${model}_int8_dynamic_range_graph.pb \
    --out_graph=${OUTPUT}/${model}_int8_freezedrange_graph.pb \
    --transforms="${TRANSFORMS2}"

    echo "${model}_int8_freezedrange_graph.pb is successfully created."
    OUTPUT_GRAPH=${OUTPUT}/${model}_int8_freezedrange_graph.pb test_ouput_graph

    # Generate the an optimized final int8 graph
    echo "Optimize the ${model} int8 frozen graph..."
    bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
    --in_graph=${OUTPUT}/${model}_int8_freezedrange_graph.pb \
    --outputs=${OUTPUT_NODES} \
    --out_graph=${OUTPUT}/${model}_int8_final_fused_graph.pb \
    --transforms="${TRANSFORMS3}"

    echo "The int8 model is successfully optimized in ${model}_int8_final_fused_graph.pb"
    OUTPUT_GRAPH=${OUTPUT}/${model}_int8_final_fused_graph.pb test_ouput_graph
}

function faster_rcnn(){
    OUTPUT_NODES='detection_boxes,detection_scores,num_detections,detection_classes'

    # Download the FP32 pre-trained model
    cd ${OUTPUT}
    wget https://storage.googleapis.com/intel-optimized-tensorflow/models/faster_rcnn_resnet50_fp32_coco_pretrained_model.tar.gz
    tar -xzvf faster_rcnn_resnet50_fp32_coco_pretrained_model.tar.gz

    cd ${TF_WORKSPACE}

    # optimize fp32 frozen graph
    bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
    --in_graph=${OUTPUT}/faster_rcnn_resnet50_fp32_coco/frozen_inference_graph.pb \
    --out_graph=${OUTPUT}/${model}_optimized_fp32_graph.pb \
    --inputs='image_tensor' \
    --outputs=${OUTPUT_NODES} \
    --transforms='strip_unused_nodes remove_nodes(op=Identity, op=CheckNumerics) fold_constants(ignore_errors=true) fold_batch_norms fold_old_batch_norms'

    # Remove downloaded pre-trained model .gz and directory
    rm -rf ${OUTPUT}/faster_rcnn_resnet50_fp32_coco
    rm -rf ${OUTPUT}/faster_rcnn_resnet50_fp32_coco_pretrained_model.tar.gz

    MODEL_NAME='FasterRCNN'
    FP32_MODEL=${OUTPUT}/${model}_optimized_fp32_graph.pb

    # to generate the logging graph
    TRANSFORMS1='insert_logging(op=RequantizationRange, show_name=true, message="__requant_min_max:")'

    # to freeze the dynamic range graph
    TRANSFORMS2='freeze_requantization_ranges(min_max_log_file="/workspace/tests/calibration_data/faster_rcnn_min_max_log.txt")'

    # to get the fused and optimized final int8 graph
    TRANSFORMS3='fuse_quantized_conv_and_requantize strip_unused_nodes'

    run_quantize_model_test
}

function rfcn(){
    OUTPUT_NODES='detection_boxes,detection_scores,num_detections,detection_classes'

    # Download the FP32 pre-trained model
    cd ${OUTPUT}
    wget https://storage.googleapis.com/intel-optimized-tensorflow/models/rfcn_resnet101_fp32_coco_pretrained_model.tar.gz
    tar -xzvf rfcn_resnet101_fp32_coco_pretrained_model.tar.gz

    # Remove the Identity ops from the FP32 frozen graph
    cd ${TF_WORKSPACE}
    bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
    --in_graph=${OUTPUT}/rfcn_resnet101_fp32_coco/frozen_inference_graph.pb \
    --out_graph=${OUTPUT}/${model}_optimized_fp32_graph.pb \
    --outputs=${OUTPUT_NODES} \
    --transforms='remove_nodes(op=Identity, op=CheckNumerics) fold_constants(ignore_errors=true)'

    # Remove downloaded pre-trained model .gz and directory
    rm -rf ${OUTPUT}/rfcn_resnet101_fp32_coco
    rm -rf ${OUTPUT}/rfcn_resnet101_fp32_coco_pretrained_model.tar.gz

    FP32_MODEL=${OUTPUT}/${model}_optimized_fp32_graph.pb
    EXTRA_ARG="--excluded_ops=ConcatV2"
    MODEL_NAME='R-FCN'

    # to generate the logging graph
    TRANSFORMS1='insert_logging(op=RequantizationRange, show_name=true, message="__requant_min_max:")'

    # to freeze the dynamic range graph
    TRANSFORMS2='freeze_requantization_ranges(min_max_log_file="/workspace/tests/calibration_data/rfcn_min_max_log.txt")'

    # to get the fused and optimized final int8 graph
    TRANSFORMS3='fuse_quantized_conv_and_requantize strip_unused_nodes'

    run_quantize_model_test
}

function resnet101(){
    OUTPUT_NODES='resnet_v1_101/SpatialSqueeze'

    # Download the FP32 pre-trained model
    cd ${OUTPUT}
    wget https://storage.googleapis.com/intel-optimized-tensorflow/models/resnet101_fp32_pretrained_model.pb
    FP32_MODEL=${OUTPUT}/resnet101_fp32_pretrained_model.pb

    # to generate the logging graph
    TRANSFORMS1='mkl_fuse_pad_and_conv'

    # to freeze the dynamic range graph
    TRANSFORMS2='freeze_requantization_ranges(min_max_log_file="/workspace/tests/calibration_data/resnet101_min_max_log.txt")'

    # to get the fused and optimized final int8 graph
    TRANSFORMS3='fuse_quantized_conv_and_requantize strip_unused_nodes'

    run_quantize_model_test
}

function resnet50(){
    OUTPUT_NODES='predict'

    # Download the FP32 pre-trained model
    cd ${OUTPUT}
    wget https://storage.googleapis.com/intel-optimized-tensorflow/models/resnet50_fp32_pretrained_model.pb
    FP32_MODEL=${OUTPUT}/resnet50_fp32_pretrained_model.pb

    # to generate the logging graph
    TRANSFORMS1='insert_logging(op=RequantizationRange, show_name=true, message="__requant_min_max:")'

    # to freeze the dynamic range graph
    TRANSFORMS2='freeze_requantization_ranges(min_max_log_file="/workspace/tests/calibration_data/resnet50_min_max_log.txt")'

    # to get the fused and optimized final int8 graph
    TRANSFORMS3='fuse_quantized_conv_and_requantize strip_unused_nodes'

    run_quantize_model_test
}

# Run all models, when new model is added append model name in alphabetical order below
for model in faster_rcnn rfcn resnet101 resnet50
do
    echo ""
    echo "Running Quantization Test for model: ${model}"
    echo ""
    echo "Initialize the test parameters for ${model} model..."
    MODEL_NAME=${model}
    ${model}
done
