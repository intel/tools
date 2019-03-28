# Quantization Tools

These tools help to transform TensorFlow graphs trained with 32-bit floating point precision to graphs with 8-bit integer precision.
This document describes how to build and use these tools.

## Prerequisites

* [Docker](https://docs.docker.com/install/) - Latest version


## Build Quantization Tools

  Build an image which contains `transform_graph` and `summarize_graph` tools.
  The initial build may take a long time, but subsequent builds will be quicker since layers are cached
   ```
        git clone https://github.com/IntelAI/tools.git
        cd tools/tensorflow_quantization

        docker build \
        --build-arg HTTP_PROXY=${HTTP_PROXY} \
        --build-arg HTTPS_PROXY=${HTTPS_PROXY} \
        --build-arg http_proxy=${http_proxy} \
        --build-arg https_proxy=${https_proxy} \
        -t quantization:latest -f Dockerfile .
   ```

## Start quantization process
  Launch quantization script `launch_quantization.py` by providing args as below,
  this will get user into container environment (`/workspace/tensorflow/`) with quantization tools.
  - `--docker-image`: Docker image tag from above step (`quantization:latest`)
  - `--pre-trained-model-dir`: Path to your pre-trained model directory,
     which will be mounted inside container at `/workspace/quantization`.
  ```
        python launch_quantization.py \
        --docker-image quantization:latest \
        --pre-trained-model-dir /home/<user>/<pre_trained_model_dir>
  ```
   Please provide the output graphs locations relative to `/workspace/quantization`, so that results are written back to local machine.

### Steps for FP32 Optimized Frozen Graph
In this section, we assume that a trained model topology graph (the model graph_def as .pb or .pbtxt file) and the checkpoint files are available.
 * The `model graph_def` is used in `step 1` to get the possible **input and output node names** of the graph.
 * Both of the `model graph_def` and the `checkpoint file` are required in `step 2` to get the **model frozen graph**.
 * The `model frozen graph`, **optimized** (based on the graph structure and operations, etc.) in `step 3`.

We also assume that you are in the TensorFlow root directory (`/workspace/tensorflow` inside the docker container) to execute the following steps.

1. Find out the possible input and output node names of the graph
    ```
        $ bazel-bin/tensorflow/tools/graph_transforms/summarize_graph \
         --in_graph=/workspace/quantization/<graph_def_file> \
         --print_structure=false >& model_nodes.txt
    ```
    In the model_nodes.txt file, look for the input and output nodes names.

2. Freeze the graph where the checkpoint values are converted into constants in the graph:
    * The `--input_graph` is the model topology graph_def, and the checkpoint file are required.
    * The `--output_node_names` are obtained from step 1.
    * Please note that the `--input_graph` can be in either binary `pb` or text `pbtxt` format,
    and the `--input_binary` flag will be enabled or disabled accordingly.
    ```
        $ python tensorflow/python/tools/freeze_graph.py \
         --input_graph /workspace/quantization/<graph_def_file> \
         --output_graph /workspace/quantization/freezed_graph.pb \
         --input_binary False \
         --input_checkpoint /workspace/quantization/<checkpoint_file> \
         --output_node_names OUTPUT_NODE_NAMES
    ```

3. Optimize the model frozen graph:
    * Set the `--in_graph` to the path of the model frozen graph (from step 2), 
    * The `--inputs` and `--outputs` are the graph input and output node names (from step 1).
    * `--transforms` to be set based on the model topology. See the TensorFlow
      [Transform Reference](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/graph_transforms#transform-reference)
      and the [Graph Transforms README](/tensorflow_quantization/graph_transforms/README.md)
      for descriptions of the different `--transforms` options.
    ```
        $ bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
         --in_graph=/workspace/quantization/freezed_graph.pb\
         --out_graph=/workspace/quantization/optimized_graph.pb \
         --inputs=INPUT_NODE_NAMES \
         --outputs=OUTPUT_NODE_NAMES \
         --transforms='fold_batch_norms'
    ```

4. Run inference using the the optimized graph `optimized_graph.pb` and check the model accuracy.
Check [Intelai/models](https://github.com/IntelAI/models) repository for TensorFlow models inference benchmarks.

### Steps for Int8 Quantization
Graph quantization to lower precision is needed for faster inference.
In this section, our objective is to quantize the output [FP32 Optimized Frozen Graph](#steps-for-fp32-optimized-frozen-graph) of the previous section.
to `Int8` precision.

5. Quantize the optimized graph (from step 3) to lower precision using the output node names (from step 1).
    ```
        $ python tensorflow/tools/quantization/quantize_graph.py \
         --input=/workspace/quantization/optimized_graph.pb \
         --output=/workspace/quantization/quantized_dynamic_range_graph.pb \
         --output_node_names=OUTPUT_NODE_NAMES \
         --mode=eightbit \
         --intel_cpu_eightbitize=True
    ```

6. Convert the quantized graph from dynamic to static re-quantization range.
   The following steps are to freeze the re-quantization range (also known as calibration):
    
    * Insert the logging op using the `insert_logging()` transform. The resulting graph (`logged_quantized_graph.pb`) from this step will be
      used in the next step to generate the min. and max. ranges for the model calibration.
        ```
        $ bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
         --in_graph=/workspace/quantization/quantized_dynamic_range_graph.pb \
         --out_graph=/workspace/quantization/logged_quantized_graph.pb \
         --transforms='insert_logging(op=RequantizationRange, show_name=true, message="__requant_min_max:")'
        ```
    
    * Generate calibration data:
        * Run inference using the `logged_quantized_graph.pb` graph that was generated in the previous step. This can be done using a
          small subset of the training dataset, since we are just running the graph to get the min and max log output.
        * The `batch_size` should be adjusted based on the data subset size.
        * During the inference run, you should see min and max output in the log that looks something like:
          ```
          ;v0/resnet_v10/conv2/conv2d/Conv2D_eightbit_requant_range__print__;__requant_min_max:[-5.75943518][3.43590856]
          ;v0/resnet_v10/conv1/conv2d/Conv2D_eightbit_requant_range__print__;__requant_min_max:[-3.63552189][5.20797968]
          ;v0/resnet_v10/conv3/conv2d/Conv2D_eightbit_requant_range__print__;__requant_min_max:[-1.44367445][1.50843954]
          ...
          ```
        * The following instructions will be referring to the log file output from your inference run as the `min_max_log.txt` file.
          For a full example of the output file might look like, see the [calibration_data](/tensorflow_quantization/tests/calibration_data) test files.
          We suggest that you store the `min_max_log.txt` in the same location specified in the [start quantization process](#start-quantization-process) section,
          which will be mounted inside the container to `/workspace/quantization`.
    
    * Replace the `RequantizationRangeOp` in the original quantized graph (from step 5)
      with the min. and max. constants using the `min_max_log.txt` file.
        ```
        $ bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
        --in_graph=/workspace/quantizationquantized_dynamic_range_graph.pb \
        --out_graph=/workspace/quantization/freezed_range_graph.pb \
        --transforms='freeze_requantization_ranges(min_max_log_file="/workspace/quantization/min_max_log.txt")'
        ```

7. Optimize the quantized graph if needed:
    * Repeat step 3 with the quantized `Int8` graph (from step 6) and a suitable `--transforms` option.
    
 
Finally, verify the quantized model performance:
 * Run inference using the final quantized graph and calculate the model accuracy.
 * Typically, the accuracy target is the optimized FP32 model accuracy values.
 * The quantized `Int8` graph accuracy should not drop more than ~0.5-1%.
    
 Check [Intelai/models](https://github.com/IntelAI/models) repository for TensorFlow models inference benchmarks with different precisions.

### Examples

* [ResNet50](https://github.com/IntelAI/models/blob/master/docs/image_recognition/quantization/Tutorial.md)
