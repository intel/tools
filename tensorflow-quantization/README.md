# Quantization Tools

These tools helps to transform TensorFlow graphs trained with 32-bit floating point precision to graphs with 8-bit integer precision.
This document describes how to build and use these tools.

## Prerequisites

* [Docker](https://docs.docker.com/install/) - Latest version


## Build Quantization Tools

* Build an image which contains `transform_graph` and `summarize_graph` tools.
  The initial build may take a long time, but subsequent builds will be quicker since layers are cached
    ```
        git clone https://github.com/NervanaSystems/tools.git
        cd tools/tensorflow-quantization

        docker build \
        --build-arg HTTP_PROXY=${HTTP_PROXY} \
        --build-arg HTTPS_PROXY=${HTTPS_PROXY} \
        --build-arg http_proxy=${http_proxy} \
        --build-arg https_proxy=${https_proxy} \
        -t quantization:latest -f Dockerfile .
    ```

## Start quantization process
* Launch quantization script `launch_quantization.py` by providing args as below,
  this will get user into container environment (`/workspace/tensorflow/`) with quantization tools.
    - `--docker-image`: Docker image tag from above step (`quantization:latest`)
    - `--pre-trained-model-dir`: Path to your pre-trained model directory,
    which will be mounted inside container at `/workspace/quantization`.


    ```
        python launch_quantization.py \
        --docker-image quantization:latest \
        --pre-trained-model-dir {path_to_pre_trained_model_dir}
    ```
    Please provide `output_path` relative to `/workspace/quantization`, so that results are written back to local machine.

**Example:** Let's perform few steps in quantization process.
* Convert fp32 frozen graph to *Optimized* fp32 frozen graph.
  - Set `--out_graph=/workspace/quantization/{name}.pb`

    ```
    root@xxxxxxxxxx:/workspace/tensorflow# bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
    > --in_graph=/workspace/quantization/ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb \
    > --out_graph=/workspace/quantization/optimized_ssd_mobilenet_fp32_graph.pb \
    > --inputs='image_tensor' \
    > --outputs='detection_boxes,detection_scores,num_detections,detection_classes' \
    > --transforms='strip_unused_nodes remove_nodes(op=Identity, op=CheckNumerics) fold_constants(ignore_errors=true) fold_batch_norms fold_old_batch_norms'
    ```
    **Output:**
    ```
    2019-03-05 22:27:10.146392: I tensorflow/tools/graph_transforms/transform_graph.cc:317] Applying strip_unused_nodes
    2019-03-05 22:27:10.243437: I tensorflow/tools/graph_transforms/transform_graph.cc:317] Applying remove_nodes
    2019-03-05 22:27:10.325350: I tensorflow/tools/graph_transforms/remove_nodes.cc:100] Skipping replacement for detection_boxes
    2019-03-05 22:27:10.325425: I tensorflow/tools/graph_transforms/remove_nodes.cc:100] Skipping replacement for detection_scores
    2019-03-05 22:27:10.325464: I tensorflow/tools/graph_transforms/remove_nodes.cc:100] Skipping replacement for num_detections
    2019-03-05 22:27:10.325508: I tensorflow/tools/graph_transforms/remove_nodes.cc:100] Skipping replacement for detection_classes
    2019-03-05 22:27:10.407018: I tensorflow/tools/graph_transforms/remove_nodes.cc:100] Skipping replacement for detection_boxes
    2019-03-05 22:27:10.407092: I tensorflow/tools/graph_transforms/remove_nodes.cc:100] Skipping replacement for detection_scores
    2019-03-05 22:27:10.407130: I tensorflow/tools/graph_transforms/remove_nodes.cc:100] Skipping replacement for num_detections
    2019-03-05 22:27:10.407173: I tensorflow/tools/graph_transforms/remove_nodes.cc:100] Skipping replacement for detection_classes
    2019-03-05 22:27:10.484257: I tensorflow/tools/graph_transforms/remove_nodes.cc:100] Skipping replacement for detection_boxes
    2019-03-05 22:27:10.484332: I tensorflow/tools/graph_transforms/remove_nodes.cc:100] Skipping replacement for detection_scores
    2019-03-05 22:27:10.484367: I tensorflow/tools/graph_transforms/remove_nodes.cc:100] Skipping replacement for num_detections
    2019-03-05 22:27:10.484391: I tensorflow/tools/graph_transforms/remove_nodes.cc:100] Skipping replacement for detection_classes
    2019-03-05 22:27:10.628321: I tensorflow/tools/graph_transforms/transform_graph.cc:317] Applying fold_constants
    2019-03-05 22:27:10.873028: I tensorflow/tools/graph_transforms/transform_graph.cc:317] Applying fold_batch_norms
    2019-03-05 22:27:10.978092: I tensorflow/tools/graph_transforms/transform_graph.cc:317] Applying fold_old_batch_norms
    ```
    Check for optimized graph inside container at `/workspace/quantization`,
    Same should be available on local machine at `--pre-trained-model-dir` path.

    ```
    root@xxxxxxxxxx:/workspace/tensorflow# ll /workspace/quantization/
    total 179816
    drwxr-xr-x 3 11570326 user      4096 Feb 27 20:37 ./
    drwxr-xr-x 1 root     root      4096 Feb 27 21:56 ../
    -rw-r--r-- 1 root     root  29075280 Feb 27 22:05 optimized_ssd_mobilenet_fp32_graph.pb
    drwxr-xr-x 3 11570326 user      4096 Feb  1  2018 ssd_mobilenet_v1_coco_2018_01_28/
    ```
* Quantize optimized fp32 frozen graph to *int8* dynamic range graph.
  - Set `--output=/workspace/quantization/{name}.pb`

    ```
    root@xxxxxxxxxx:/workspace/tensorflow# python tensorflow/tools/quantization/quantize_graph.py \
    > --input=/workspace/quantization/optimized_ssd_mobilenet_fp32_graph.pb \
    > --output=/workspace/quantization/int8_dynamic_range_ssd_mobilenet_graph.pb \
    > --output_node_names='detection_boxes,detection_scores,num_detections,detection_classes' \
    > --mode=eightbit \
    > --intel_cpu_eightbitize=True
    ```
    **Output:**
    ```
    W0305 22:29:12.134598 140044576818944 deprecation.py:323] From tensorflow/tools/quantization/quantize_graph.py:482: remove_training_nodes (from tensorflow.python.framework.graph_util_impl) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.compat.v1.graph_util.remove_training_nodes
    2019-03-05 22:29:13.706089: I tensorflow/core/platform/cpu_feature_guard.cc:145] This TensorFlow binary is optimized with Intel(R) MKL-DNN to use the following CPU instructions in performance critical operations:  AVX512F
    To enable them in non-MKL-DNN operations, rebuild TensorFlow with the appropriate compiler flags.
    2019-03-05 22:29:13.749701: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2500000000 Hz
    2019-03-05 22:29:13.761915: I tensorflow/compiler/xla/service/service.cc:162] XLA service 0x8c306c0 executing computations on platform Host. Devices:
    2019-03-05 22:29:13.761968: I tensorflow/compiler/xla/service/service.cc:169]   StreamExecutor device (0): <undefined>, <undefined>
    2019-03-05 22:29:13.771628: I tensorflow/core/common_runtime/process_util.cc:92] Creating new thread pool with default inter op setting: 2. Tune using inter_op_parallelism_threads for best performance.
    W0305 22:29:13.772703 140044576818944 deprecation.py:323] From tensorflow/tools/quantization/quantize_graph.py:356: quantize_v2 (from tensorflow.python.ops.array_ops) is deprecated and will be removed after 2017-10-25.
    Instructions for updating:
    `tf.quantize_v2` is deprecated, please use `tf.quantization.quantize` instead.
    W0305 22:29:19.609273 140044576818944 deprecation.py:323] From tensorflow/tools/quantization/quantize_graph.py:1556: extract_sub_graph (from tensorflow.python.framework.graph_util_impl) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.compat.v1.graph_util.extract_sub_graph
    W0305 22:29:20.707792 140044576818944 deprecation.py:323] From tensorflow/tools/quantization/quantize_graph.py:1665: __init__ (from tensorflow.python.platform.gfile) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.gfile.GFile.
    ```
    Check for optimized graph inside container at `/workspace/quantization`,
    Same should be available on local machine at `--pre-trained-model-dir` path.

    ```
    root@xxxxxxxxxx:/workspace/tensorflow# ll /workspace/quantization/
    total 179816
    drwxr-xr-x 3 11570326 user      4096 Feb 27 22:19 ./
    drwxr-xr-x 1 root     root      4096 Feb 27 21:56 ../
    -rw-r--r-- 1 root     root   8997297 Feb 27 22:19 int8_dynamic_range_ssd_mobilenet_graph.pb
    -rw-r--r-- 1 root     root  29075280 Feb 27 22:05 optimized_ssd_mobilenet_fp32_graph.pb
    drwxr-xr-x 3 11570326 user      4096 Feb  1  2018 ssd_mobilenet_v1_coco_2018_01_28/
    ```
### Follow below instructions as per model

* [Resnet50]
