#!/usr/bin/env bash
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

set -e
set -x

echo 'Running with parameters:'
echo "    WORKSPACE: ${WORKSPACE}"
echo "    TF_WORKSPACE: ${TF_WORKSPACE}"
echo "    TEST_WORKSPACE: ${TEST_WORKSPACE}"
echo "    Mounted Volumes:"
echo "        ${PRE_TRAINED_MODEL_DIR} mounted on: ${MOUNT_OUTPUT}"

# Intel Models GCS bucket URL
INTEL_MODELS_BUCKET="https://storage.googleapis.com/intel-optimized-tensorflow/models"

# output directory for tests
OUTPUT=${MOUNT_OUTPUT}/output

# mounted datasets directory
DATASET=${MOUNT_OUTPUT}/dataset

function test_output_graph(){
    test -f ${OUTPUT_GRAPH}
    if [ $? == 1 ]; then
        # clean up the output directory if the test fails.
        rm -rf ${OUTPUT}
        exit $?
    fi
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

    OUTPUT_GRAPH=${OUTPUT}/${model}_int8_dynamic_range_graph.pb test_output_graph
    echo ""
    echo "${model}_int8_dynamic_range_graph.pb is successfully created."
    echo ""

    if [ ${model}=="rfcn" ] || [ ${model}=="resnet101" ]; then
        # Apply Pad Fusion optimization:
        echo "Apply Pad Fusion optimization for the int8 dynamic range ${model} graph..."
        bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
        --in_graph=${OUTPUT}/${model}_int8_dynamic_range_graph.pb \
        --out_graph=${OUTPUT}/${model}_int8_dynamic_range_graph.pb \
        --outputs=${OUTPUT_NODES} \
        --transforms='mkl_fuse_pad_and_conv'

        OUTPUT_GRAPH=${OUTPUT}/${model}_int8_dynamic_range_graph.pb test_output_graph
    fi

    # Generate graph with logging
    echo "Generate the graph with logging for ${model} model..."
    bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
    --in_graph=${OUTPUT}/${model}_int8_dynamic_range_graph.pb \
    --out_graph=${OUTPUT}/${model}_int8_logged_graph.pb \
    --transforms="${TRANSFORMS1}"

    OUTPUT_GRAPH=${OUTPUT}/${model}_int8_logged_graph.pb test_output_graph
    echo ""
    echo "${model}_int8_logged_graph.pb is successfully created."
    echo ""

    # Model Calibration: Generate the model min_max_log.txt file
    echo "Generate ${model} min_max_log.txt file ..."
    if [ ${model} == "inceptionv3" ] || [ ${model} == "inceptionv4" ] || [ ${model} == "inception_resnet_v2" ] || [ ${model} == "resnet101" ]; then
        calibrate_image_recognition_common
    elif [ ${model} == "faster_rcnn" ] || [ ${model} == "rfcn" ]; then
        calibrate_object_detection_common
    else
        calibrate_${model}
    fi

    # Convert the dynamic range int8 graph to freezed range graph
    cd ${TF_WORKSPACE}
    echo "Freeze the dynamic range graph using the min max constants from ${model}_min_max_log.txt..."
    bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
    --in_graph=${OUTPUT}/${model}_int8_dynamic_range_graph.pb \
    --out_graph=${OUTPUT}/${model}_int8_freezedrange_graph.pb \
    --transforms="${TRANSFORMS2}"

    OUTPUT_GRAPH=${OUTPUT}/${model}_int8_freezedrange_graph.pb test_output_graph
    echo ""
    echo "${model}_int8_freezedrange_graph.pb is successfully created."
    echo ""

    # Generate the an optimized final int8 graph
    echo "Optimize the ${model} int8 frozen graph..."
    bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
    --in_graph=${OUTPUT}/${model}_int8_freezedrange_graph.pb \
    --outputs=${OUTPUT_NODES} \
    --out_graph=${OUTPUT}/${model}_int8_final_fused_graph.pb \
    --transforms="${TRANSFORMS3}"

    OUTPUT_GRAPH=${OUTPUT}/${model}_int8_final_fused_graph.pb test_output_graph
    echo ""
    echo "The ${model} int8 model is successfully optimized in ${model}_int8_final_fused_graph.pb"
    echo ""
}

function faster_rcnn(){
    OUTPUT_NODES='detection_boxes,detection_scores,num_detections,detection_classes'

    # Download the FP32 pre-trained model
    cd ${OUTPUT}
    FP32_MODEL_DIR="faster_rcnn_resnet50_fp32_coco"
    FP32_MODEL_TAR="faster_rcnn_resnet50_fp32_coco_pretrained_model.tar.gz"

    wget -q ${INTEL_MODELS_BUCKET}/${FP32_MODEL_TAR}
    tar -xzvf ${FP32_MODEL_TAR}
    FP32_MODEL=${OUTPUT}/${FP32_MODEL_DIR}/frozen_inference_graph.pb

    cd ${TF_WORKSPACE}

    # optimize fp32 frozen graph
    bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
    --in_graph=${FP32_MODEL} \
    --out_graph=${OUTPUT}/${model}_optimized_fp32_graph.pb \
    --inputs='image_tensor' \
    --outputs=${OUTPUT_NODES} \
    --transforms='strip_unused_nodes remove_nodes(op=Identity, op=CheckNumerics) fold_constants(ignore_errors=true) fold_batch_norms fold_old_batch_norms'

    # Remove downloaded pre-trained model .gz and directory
    rm -rf ${OUTPUT}/${FP32_MODEL_DIR}
    rm -rf ${OUTPUT}/${FP32_MODEL_TAR}

    MODEL_NAME='FasterRCNN'
    FP32_MODEL=${OUTPUT}/${model}_optimized_fp32_graph.pb

    # to generate the logging graph
    TRANSFORMS1='insert_logging(op=RequantizationRange, show_name=true, message="__requant_min_max:")'

    # to freeze the dynamic range graph
    TRANSFORMS2='freeze_requantization_ranges(min_max_log_file="/workspace/mounted_dir/output/faster_rcnn_min_max_log.txt")'

    # to get the fused and optimized final int8 graph
    TRANSFORMS3='fuse_quantized_conv_and_requantize strip_unused_nodes'

    run_quantize_model_test
}

function inceptionv3() {
    OUTPUT_NODES='predict'

    # Download the FP32 pre-trained model
    cd ${OUTPUT}
    FP32_MODEL="inceptionv3_fp32_pretrained_model.pb"
    wget -q ${INTEL_MODELS_BUCKET}/${FP32_MODEL}
    FP32_MODEL=${OUTPUT}/${FP32_MODEL}

    EXTRA_ARG="--excluded_ops=MaxPool,AvgPool,ConcatV2"

    # to generate the logging graph
    TRANSFORMS1='insert_logging(op=RequantizationRange, show_name=true, message="__requant_min_max:")'

    # to freeze the dynamic range graph
    TRANSFORMS2='freeze_requantization_ranges(min_max_log_file="/workspace/mounted_dir/output/inceptionv3_min_max_log.txt")'

    # to get the fused and optimized final int8 graph
    TRANSFORMS3='fuse_quantized_conv_and_requantize strip_unused_nodes'

    run_quantize_model_test

    # to rerange quantize concat
    TRANSFORMS4='rerange_quantized_concat'

    # run fourth transform separately since run_quantize_model_test just runs three
    bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
    --in_graph=${OUTPUT}/${model}_int8_final_fused_graph.pb \
    --outputs=${OUTPUT_NODES} \
    --out_graph=${OUTPUT}/${model}_int8_final_graph.pb \
    --transforms="${TRANSFORMS4}" \
    --output_as_text=false

    OUTPUT_GRAPH=${OUTPUT}/${model}_int8_final_graph.pb test_output_graph
}

function inceptionv4() {
    OUTPUT_NODES='InceptionV4/Logits/Predictions'

    # Download the FP32 pre-trained model
    cd ${OUTPUT}
    FP32_MODEL="inceptionv4_fp32_pretrained_model.pb"
    wget -q ${INTEL_MODELS_BUCKET}/${FP32_MODEL}
    FP32_MODEL=${OUTPUT}/${FP32_MODEL}

    # to generate the logging graph
    TRANSFORMS1='insert_logging(op=RequantizationRange, show_name=true, message="__requant_min_max:")'

    # to freeze the dynamic range graph
    TRANSFORMS2='freeze_requantization_ranges(min_max_log_file="/workspace/mounted_dir/output/inceptionv4_min_max_log.txt")'

    # to get the fused and optimized final int8 graph
    TRANSFORMS3='fuse_quantized_conv_and_requantize strip_unused_nodes'

    run_quantize_model_test

    # to rerange quantize concat
    TRANSFORMS4='rerange_quantized_concat'

    # run fourth transform separately since run_quantize_model_test just runs three
    bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
    --in_graph=${OUTPUT}/${model}_int8_final_fused_graph.pb \
    --outputs=${OUTPUT_NODES} \
    --out_graph=${OUTPUT}/${model}_int8_final_graph.pb \
    --transforms="${TRANSFORMS4}"

    OUTPUT_GRAPH=${OUTPUT}/${model}_int8_final_graph.pb test_output_graph
}

function inception_resnet_v2() {
    OUTPUT_NODES='InceptionResnetV2/Logits/Predictions'

    # Download the FP32 pre-trained model
    cd ${OUTPUT}
    FP32_MODEL="inception_resnet_v2_fp32_pretrained_model.pb"
    wget -q ${INTEL_MODELS_BUCKET}/${FP32_MODEL}
    FP32_MODEL=${OUTPUT}/${FP32_MODEL}

    EXTRA_ARG="--excluded_ops=MaxPool,AvgPool,ConcatV2"
    # to generate the logging graph
    TRANSFORMS1='insert_logging(op=RequantizationRange, show_name=true, message="__requant_min_max:")'

    # to freeze the dynamic range graph
    TRANSFORMS2='freeze_requantization_ranges(min_max_log_file="/workspace/mounted_dir/output/inception_resnet_v2_min_max_log.txt")'

    # to rerange quantize concat and get the fused optimized final int8 graph
    TRANSFORMS3='rerange_quantized_concat fuse_quantized_conv_and_requantize strip_unused_nodes'

    run_quantize_model_test
}

function rfcn(){
    OUTPUT_NODES='detection_boxes,detection_scores,num_detections,detection_classes'

    # Download the FP32 pre-trained model
    cd ${OUTPUT}
    FP32_MODEL_DIR="rfcn_resnet101_coco_2018_01_28"
    FP32_MODEL_TAR="rfcn_resnet101_fp32_coco_pretrained_model.tar.gz"
    wget -q ${INTEL_MODELS_BUCKET}/${FP32_MODEL_TAR}
    tar -xzvf ${FP32_MODEL_TAR}
    FP32_MODEL=${OUTPUT}/${FP32_MODEL_DIR}/frozen_inference_graph.pb

    # Remove the Identity ops from the FP32 frozen graph
    cd ${TF_WORKSPACE}
    bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
    --in_graph=${FP32_MODEL} \
    --out_graph=${OUTPUT}/${model}_optimized_fp32_graph.pb \
    --outputs=${OUTPUT_NODES} \
    --transforms='remove_nodes(op=Identity, op=CheckNumerics) fold_constants(ignore_errors=true)'

    # Remove downloaded pre-trained model .gz and directory
    rm -rf ${OUTPUT}/${FP32_MODEL_DIR}
    rm -rf ${OUTPUT}/${FP32_MODEL_TAR}

    FP32_MODEL=${OUTPUT}/${model}_optimized_fp32_graph.pb
    EXTRA_ARG="--excluded_ops=ConcatV2"
    MODEL_NAME='R-FCN'

    # to generate the logging graph
    TRANSFORMS1='insert_logging(op=RequantizationRange, show_name=true, message="__requant_min_max:")'

    # to freeze the dynamic range graph
    TRANSFORMS2='freeze_requantization_ranges(min_max_log_file="/workspace/mounted_dir/output/rfcn_min_max_log.txt")'

    # to get the fused and optimized final int8 graph
    TRANSFORMS3='fuse_quantized_conv_and_requantize strip_unused_nodes'

    run_quantize_model_test
}

function resnet101(){
    OUTPUT_NODES='resnet_v1_101/predictions/Reshape_1'

    # Download the FP32 pre-trained model
    cd ${OUTPUT}
    FP32_MODEL="resnet101_fp32_pretrained_model.pb"
    wget -q ${INTEL_MODELS_BUCKET}/${FP32_MODEL}
    FP32_MODEL=${OUTPUT}/${FP32_MODEL}

    # to generate the logging graph
    TRANSFORMS1='insert_logging(op=RequantizationRange, show_name=true, message="__requant_min_max:")'

    # to freeze the dynamic range graph
    TRANSFORMS2='freeze_requantization_ranges(min_max_log_file="/workspace/mounted_dir/output/resnet101_min_max_log.txt")'

    # to get the fused and optimized final int8 graph
    TRANSFORMS3='fuse_quantized_conv_and_requantize strip_unused_nodes'

    run_quantize_model_test
}

function resnet50(){
    OUTPUT_NODES='predict'

    # Download the FP32 pre-trained model
    cd ${OUTPUT}
    FP32_MODEL="resnet50_fp32_pretrained_model.pb"
    wget -q ${INTEL_MODELS_BUCKET}/${FP32_MODEL}
    FP32_MODEL=${OUTPUT}/${FP32_MODEL}

    # to generate the logging graph
    TRANSFORMS1='insert_logging(op=RequantizationRange, show_name=true, message="__requant_min_max:")'

    # to freeze the dynamic range graph
    TRANSFORMS2='freeze_requantization_ranges(min_max_log_file="/workspace/mounted_dir/output/resnet50_min_max_log.txt")'

    # to get the fused and optimized final int8 graph
    TRANSFORMS3='fuse_quantized_conv_and_requantize strip_unused_nodes'

    run_quantize_model_test
}

function resnet50v1_5(){
    OUTPUT_NODES='ArgMax,softmax_tensor'

    # Download and optimize the FP32 pre-trained model
    cd ${OUTPUT}
    wget -q https://zenodo.org/record/2535873/files/resnet50_v1.pb
    FP32_MODEL="resnet50_v1.pb"
    FP32_MODEL=${OUTPUT}/${FP32_MODEL}

    cd ${TF_WORKSPACE}

    # optimize fp32 frozen graph
    bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
    --in_graph=${FP32_MODEL} \
    --out_graph=${OUTPUT}/${model}_optimized_fp32_graph.pb \
    --inputs='input_tensor' \
    --outputs=${OUTPUT_NODES} \
    --transforms='strip_unused_nodes remove_nodes(op=Identity, op=CheckNumerics) fold_batch_norms fold_old_batch_norms'

    rm ${FP32_MODEL}
    FP32_MODEL=${OUTPUT}/${model}_optimized_fp32_graph.pb

    EXTRA_ARG="--per_channel=True"

    # to generate the logging graph
    TRANSFORMS1='insert_logging(op=RequantizationRange, show_name=true, message="__requant_min_max:")'

    # to freeze the dynamic range graph
    TRANSFORMS2='freeze_requantization_ranges(min_max_log_file="/workspace/mounted_dir/output/resnet50v1_5_min_max_log.txt")'

    # to get the fused and optimized final int8 graph
    TRANSFORMS3='fuse_quantized_conv_and_requantize strip_unused_nodes'

    run_quantize_model_test
}

function ssd_mobilenet(){
    OUTPUT_NODES='detection_boxes,detection_scores,num_detections,detection_classes'

    # Download the FP32 pre-trained model
    cd ${OUTPUT}
    FP32_MODEL_DIR="ssd_mobilenet_v1_coco_2018_01_28"
    FP32_MODEL_TAR="ssd_mobilenet_v1_coco_2018_01_28.tar.gz"
    wget -q http://download.tensorflow.org/models/object_detection/${FP32_MODEL_TAR}
    tar -xzvf ${FP32_MODEL_TAR}
    FP32_MODEL=${OUTPUT}/${FP32_MODEL_DIR}/frozen_inference_graph.pb

    cd ${TF_WORKSPACE}

    # optimize fp32 frozen graph
    bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
    --in_graph=${FP32_MODEL} \
    --out_graph=${OUTPUT}/${model}_optimized_fp32_graph.pb \
    --inputs='image_tensor' \
    --outputs=${OUTPUT_NODES} \
    --transforms='strip_unused_nodes remove_nodes(op=Identity, op=CheckNumerics) fold_constants(ignore_errors=true) fold_batch_norms fold_old_batch_norms'

    # Remove downloaded pre-trained model .gz and directory
    rm -rf ${OUTPUT}/${FP32_MODEL_DIR}
    rm -rf ${OUTPUT}/${FP32_MODEL_TAR}

    FP32_MODEL=${OUTPUT}/${model}_optimized_fp32_graph.pb

    # to generate the logging graph
    TRANSFORMS1='insert_logging(op=RequantizationRange, show_name=true, message="__requant_min_max:")'

    # to freeze the dynamic range graph
    TRANSFORMS2='freeze_requantization_ranges(min_max_log_file="/workspace/mounted_dir/output/ssd_mobilenet_min_max_log.txt")'

    # to get the fused and optimized final int8 graph
    TRANSFORMS3='fuse_quantized_conv_and_requantize strip_unused_nodes'

    run_quantize_model_test
}

function ssd_resnet34() {
    OUTPUT_NODES='v/stack,v/Softmax'

    # Download the FP32 pre-trained model
    cd ${OUTPUT}
    FP32_MODEL="ssd_resnet34_fp32_bs1_pretrained_model.pb"
    wget -q ${INTEL_MODELS_BUCKET}/${FP32_MODEL}
    FP32_MODEL=${OUTPUT}/${FP32_MODEL}

    # to generate the logging graph
    TRANSFORMS1='insert_logging(op=RequantizationRange, show_name=true, message="__requant_min_max:")'

    # to freeze the dynamic range graph
    TRANSFORMS2='freeze_requantization_ranges(min_max_log_file="/workspace/mounted_dir/output/ssd_resnet34_min_max_log.txt")'

    # to get the fused and optimized final int8 graph
    TRANSFORMS3='fuse_quantized_conv_and_requantize strip_unused_nodes'

    run_quantize_model_test
}

function ssd_vgg16(){
    OUTPUT_NODES='ExpandDims,ExpandDims_1,ExpandDims_2,ExpandDims_3,ExpandDims_4,ExpandDims_5,ExpandDims_6,ExpandDims_7,ExpandDims_8,ExpandDims_9,ExpandDims_10,ExpandDims_11,ExpandDims_12,ExpandDims_13,ExpandDims_14,ExpandDims_15,ExpandDims_16,ExpandDims_17,ExpandDims_18,ExpandDims_19,ExpandDims_20,ExpandDims_21,ExpandDims_22,ExpandDims_23,ExpandDims_24,ExpandDims_25,ExpandDims_26,ExpandDims_27,ExpandDims_28,ExpandDims_29,ExpandDims_30,ExpandDims_31,ExpandDims_32,ExpandDims_33,ExpandDims_34,ExpandDims_35,ExpandDims_36,ExpandDims_37,ExpandDims_38,ExpandDims_39,ExpandDims_40,ExpandDims_41,ExpandDims_42,ExpandDims_43,ExpandDims_44,ExpandDims_45,ExpandDims_46,ExpandDims_47,ExpandDims_48,ExpandDims_49,ExpandDims_50,ExpandDims_51,ExpandDims_52,ExpandDims_53,ExpandDims_54,ExpandDims_55,ExpandDims_56,ExpandDims_57,ExpandDims_58,ExpandDims_59,ExpandDims_60,ExpandDims_61,ExpandDims_62,ExpandDims_63,ExpandDims_64,ExpandDims_65,ExpandDims_66,ExpandDims_67,ExpandDims_68,ExpandDims_69,ExpandDims_70,ExpandDims_71,ExpandDims_72,ExpandDims_73,ExpandDims_74,ExpandDims_75,ExpandDims_76,ExpandDims_77,ExpandDims_78,ExpandDims_79,ExpandDims_80,ExpandDims_81,ExpandDims_82,ExpandDims_83,ExpandDims_84,ExpandDims_85,ExpandDims_86,ExpandDims_87,ExpandDims_88,ExpandDims_89,ExpandDims_90,ExpandDims_91,ExpandDims_92,ExpandDims_93,ExpandDims_94,ExpandDims_95,ExpandDims_96,ExpandDims_97,ExpandDims_98,ExpandDims_99,ExpandDims_100,ExpandDims_101,ExpandDims_102,ExpandDims_103,ExpandDims_104,ExpandDims_105,ExpandDims_106,ExpandDims_107,ExpandDims_108,ExpandDims_109,ExpandDims_110,ExpandDims_111,ExpandDims_112,ExpandDims_113,ExpandDims_114,ExpandDims_115,ExpandDims_116,ExpandDims_117,ExpandDims_118,ExpandDims_119,ExpandDims_120,ExpandDims_121,ExpandDims_122,ExpandDims_123,ExpandDims_124,ExpandDims_125,ExpandDims_126,ExpandDims_127,ExpandDims_128,ExpandDims_129,ExpandDims_130,ExpandDims_131,ExpandDims_132,ExpandDims_133,ExpandDims_134,ExpandDims_135,ExpandDims_136,ExpandDims_137,ExpandDims_138,ExpandDims_139,ExpandDims_140,ExpandDims_141,ExpandDims_142,ExpandDims_143,ExpandDims_144,ExpandDims_145,ExpandDims_146,ExpandDims_147,ExpandDims_148,ExpandDims_149,ExpandDims_150,ExpandDims_151,ExpandDims_152,ExpandDims_153,ExpandDims_154,ExpandDims_155,ExpandDims_156,ExpandDims_157,ExpandDims_158,ExpandDims_159'

    # Download the FP32 pre-trained model
    cd ${OUTPUT}
    FP32_MODEL="ssdvgg16_fp32_pretrained_model.pb"
    wget -q ${INTEL_MODELS_BUCKET}/${FP32_MODEL}
    FP32_MODEL=${OUTPUT}/${FP32_MODEL}
    EXTRA_ARG="--excluded_ops=ConcatV2 --output_binary=True"

    # to generate the logging graph
    TRANSFORMS1='insert_logging(op=RequantizationRange, show_name=true, message="__requant_min_max:")'

    # to freeze the dynamic range graph
    TRANSFORMS2='freeze_requantization_ranges(min_max_log_file="/workspace/mounted_dir/output/ssd_vgg16_min_max_log.txt")'

    # to get the fused and optimized final int8 graph
    TRANSFORMS3='fuse_quantized_conv_and_requantize strip_unused_nodes'

    run_quantize_model_test
}

######### Model Calibration #######

function get_cocoapi() {
  # get arg for where the cocoapi repo was cloned
  cocoapi_dir=${1}

  # get arg for the location where we want the pycocotools
  parent_dir=${2}
  pycocotools_dir=${parent_dir}/pycocotools

  # If pycoco tools aren't already found, then builds the coco python API
  if [ ! -d ${pycocotools_dir} ]; then
    # This requires that the cocoapi is cloned in the external model source dir
    if [ -d "${cocoapi_dir}/PythonAPI" ]; then
      # install cocoapi
      pushd ${cocoapi_dir}/PythonAPI
      echo "Installing COCO API"
      make
      cp -r pycocotools ${parent_dir}
      popd
    else
      echo "${cocoapi_dir}/PythonAPI directory was not found"
      echo "Unable to install the python cocoapi."
      exit 1
    fi
  else
    echo "pycocotools were found at: ${pycocotools_dir}"
  fi
}

function install_protoc() {
  pushd "${TENSORFLOW_MODELS}/models/research"

  # install protoc, if necessary, then compile protoc files
  if [ ! -f "bin/protoc" ]; then
    install_location=$1
    echo "protoc not found, installing protoc from ${install_location}"
    apt-get -y install wget
    wget -O protobuf.zip ${install_location}
    unzip -o protobuf.zip
    rm protobuf.zip
  else
    echo "protoc already found"
  fi

  echo "Compiling protoc files"
  ./bin/protoc object_detection/protos/*.proto --python_out=.
  popd
}

# run inference using the logged graph to generate the min_max ranges file.
function generate_min_max_ranges(){
    INTEL_MODELS=${OUTPUT}/models

    if [ ! -d ${INTEL_MODELS} ]; then
        echo "Intel Models directory cannot be found in ${INTEL_MODELS}."
        exit 1
    fi
    cd ${INTEL_MODELS}/benchmarks

    ## install common dependencies
    apt update
    apt full-upgrade -y
    apt-get install python-tk numactl -y
    apt install -y libsm6 libxext6
    pip install requests

    if [ ${model} == "ssd_vgg16" ]; then
        get_cocoapi ${SSD_TENSORFLOW}/coco ${INTEL_MODELS}/models/object_detection/tensorflow/ssd_vgg16/inference/
    fi

    if [ ${model} == "faster_rcnn" ] || [ ${model} == "rfcn" ] || [ ${model} == "ssd_mobilenet" ]; then
        # install dependencies
        pip install -r "${INTEL_MODELS}/benchmarks/object_detection/tensorflow/faster_rcnn/requirements.txt"

        cd "${TENSORFLOW_MODELS}/models/research"
        # install protoc v3.3.0, if necessary, then compile protoc files
        install_protoc "https://github.com/google/protobuf/releases/download/v3.3.0/protoc-3.3.0-linux-x86_64.zip"

        # install cocoapi
        get_cocoapi ${TENSORFLOW_MODELS}/models/cocoapi ${TENSORFLOW_MODELS}/models/research/

        if [ ${model} == "faster_rcnn" ]; then
            chmod +x ${INTEL_MODELS}/models/object_detection/tensorflow/faster_rcnn/inference/int8/coco_int8.sh
        elif [ ${model} == "ssd_mobilenet" ]; then
            chmod +x ${INTEL_MODELS}/models/object_detection/tensorflow/ssd-mobilenet/inference/int8/coco_int8.sh
        else
            chmod +x ${INTEL_MODELS}/models/object_detection/tensorflow/rfcn/inference/int8/coco_mAP.sh
        fi
    fi

    if [ ${model} == "ssd_resnet34" ]; then
        for line in $(cat ${INTEL_MODELS}/benchmarks/object_detection/tensorflow/ssd-resnet34/requirements.txt)
        do
          pip install $line
        done
        apt install -y git-all
        old_dir=${PWD}
        cd /tmp
        git clone --single-branch https://github.com/tensorflow/benchmarks.git
        cd benchmarks
        git checkout 1e7d788042dfc6d5e5cd87410c57d5eccee5c664
        cd ${old_dir}
    fi

    # run inference
    if [ ${model} == "resnet50" ]; then
        cd ${INTEL_MODELS}/benchmarks
        python launch_benchmark.py \
        --mode inference \
        --model-name ${model} \
        --precision int8 \
        --framework tensorflow \
        --in-graph ${FP32_MODEL} \
        --accuracy-only ${CALIBRATE_ARGS}

        mkdir dataset && mv calibration-1-of-1 dataset
    fi

    cd ${INTEL_MODELS}/benchmarks
    timeout 100s python launch_benchmark.py \
    --mode inference \
    --precision int8 \
    --framework tensorflow \
    --in-graph ${OUTPUT}/${model}_int8_logged_graph.pb \
    --accuracy-only ${MODEL_ARG} >& ${OUTPUT}/${model}_min_max_log.txt || continue
}

function calibrate_image_recognition_common() {
    # for models: inceptionv3, inceptionv4, inception_resnet_v2, and resnet101
    MODEL_ARG="--batch-size 100 --data-location ${DATASET}/imagenet-data --model-name ${model}"
    if [ ${model} == "inceptionv3" ] || [ ${model} == "resnet101" ]; then
        MODEL_ARG="${MODEL_ARG} -- calibration_only=True"
    fi
    generate_min_max_ranges
}

function calibrate_object_detection_common(){
    TENSORFLOW_MODELS=${OUTPUT}/tensorflow_models

    if [ -d ${TENSORFLOW_MODELS} ]; then
        rm -rf ${TENSORFLOW_MODELS}
    fi
    cd ${OUTPUT}
    mkdir tensorflow_models && cd tensorflow_models
    git clone https://github.com/tensorflow/models.git
    cd ${TENSORFLOW_MODELS}/models
    git clone https://github.com/cocodataset/cocoapi.git

    cd research/object_detection
    chmod 777 metrics
    cd "metrics"
    chmod 777 offline_eval_map_corloc.py
    sed -i.bak 162s/eval_input_config/eval_input_configs/ offline_eval_map_corloc.py
    sed -i.bak 91s/input_config/input_config[0]/ offline_eval_map_corloc.py
    sed -i.bak 92s/input_config/input_config[0]/ offline_eval_map_corloc.py
    sed -i.bak 95s/input_config/input_config[0]/ offline_eval_map_corloc.py

    cp ${DATASET}/coco-data/coco_train.record ${DATASET}/coco_val.record

    if [ ${model} == "faster_rcnn" ]; then
        MODEL_ARG="--model-source-dir ${TENSORFLOW_MODELS}/models --data-location ${DATASET}/coco_val.record --model-name ${model}"
    elif [ ${model} == "rfcn" ]; then
        MODEL_ARG="--model-source-dir ${TENSORFLOW_MODELS}/models --data-location ${DATASET}/coco_val.record --model-name ${model} -- split="accuracy_message""
    fi
    generate_min_max_ranges
}

function calibrate_resnet50() {
    CALIBRATE_ARGS="--batch-size 100 --data-location ${DATASET}/imagenet-data --model-name ${model} -- calibration_only=True"

    MODEL_ARG="--batch-size 100 --data-location ${OUTPUT}/models/benchmarks/dataset --model-name ${model} -- calibrate=True"
    generate_min_max_ranges
}

function calibrate_resnet50v1_5() {
    MODEL_ARG="--batch-size 1 --data-location ${DATASET}/imagenet-data --model-name ${model} --warmup_steps=100"
    generate_min_max_ranges
}

function calibrate_ssd_mobilenet() {
    TENSORFLOW_MODELS=${OUTPUT}/tensorflow_models

    if [ -d ${TENSORFLOW_MODELS} ]; then
        rm -rf ${TENSORFLOW_MODELS}
    fi
    cd ${OUTPUT}
    mkdir tensorflow_models && cd tensorflow_models
    git clone https://github.com/tensorflow/models.git
    
    cd ${TENSORFLOW_MODELS}/models
    git checkout 20da786b078c85af57a4c88904f7889139739ab0
    git clone https://github.com/cocodataset/cocoapi.git

    chmod -R 777 ${TENSORFLOW_MODELS}/models/research/object_detection/inference/detection_inference.py
    sed -i.bak "s/'r'/'rb'/g" ${TENSORFLOW_MODELS}/models/research/object_detection/inference/detection_inference.py

    MODEL_ARG="--batch-size 1 --model-source-dir ${TENSORFLOW_MODELS}/models --data-location ${DATASET}/coco-data/coco_train.record --model-name ssd-mobilenet"
    generate_min_max_ranges
}

function calibrate_ssd_resnet34() {
    TENSORFLOW_MODELS=${OUTPUT}/tensorflow_models

    if [ -d ${TENSORFLOW_MODELS} ]; then
        rm -rf ${TENSORFLOW_MODELS}
    fi
    cd ${OUTPUT}
    mkdir tensorflow_models && cd tensorflow_models
    git clone https://github.com/tensorflow/models.git

    cd ${TENSORFLOW_MODELS}/models
    git checkout f505cecde2d8ebf6fe15f40fb8bc350b2b1ed5dc
    git clone https://github.com/cocodataset/cocoapi.git
    cp ${DATASET}/coco-data/coco_val.record ${DATASET}/coco-data/validation-00000-of-00001

    MODEL_ARG="--batch-size 1 --model-source-dir ${TENSORFLOW_MODELS}/models --data-location ${DATASET}/coco-data --model-name ssd-resnet34"
    generate_min_max_ranges
}

function calibrate_ssd_vgg16() {
    SSD_TENSORFLOW=${OUTPUT}/SSD.TensorFlow
    if [ ! -d ${SSD_TENSORFLOW} ]; then
        cd ${OUTPUT}
        git clone https://github.com/HiKapok/SSD.TensorFlow.git
        cd SSD.TensorFlow
        git checkout 2d8b0cb9b2e70281bf9dce438ff17ffa5e59075c
        git clone https://github.com/waleedka/coco.git
    fi

    # install dependencies
    pip install opencv-python Cython

    MODEL_ARG="--batch-size 1 --model-source-dir ${OUTPUT}/SSD.TensorFlow --data-location ${DATASET}/coco-data-ssdvgg16 --model-name ${model}"
    generate_min_max_ranges
}
########################

# Run all models, when new model is added append model name in alphabetical order below

for model in faster_rcnn inceptionv3 inceptionv4 inception_resnet_v2 rfcn resnet101 resnet50 resnet50v1_5 ssd_vgg16 ssd_mobilenet ssd_resnet34
do
    echo ""
    echo "Running Quantization Test for model: ${model}"
    echo ""
    echo "Initialize the test parameters for ${model} model..."
    MODEL_NAME=${model}
    ${model}
done
