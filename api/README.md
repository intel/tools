# Quantization Python Programming API Quick Start

Content:
- [Quantization Python Programming API Quick Start](#quantization-python-programming-api-quick-start)
  - [Goal](#goal)
  - [Prerequisites](#prerequisites)
  - [Step-by-step Procedure for ResNet-50 Quantization](#step-by-step-procedure-for-resnet-50-quantization)
  - [Integration with Model Zoo for Intel Architecture](#integration-with-model-zoo-for-intel-architecture)
  - [Tools](#tools)
    - [Summarize graph](#summarize-graph)
  - [Docker support](#docker-support)
  - [FAQ](#faq)


## Goal

The Quantization Python programming API is to:
* Unify the quantization tools calling entry, 
* Remove the Tensorflow source build dependency,
* Transparent the model quantization process, 
* Reduce the quantization steps,
* Seamlessly adapt to inference with Python script.

This feature is under active development, and more intelligent features will come in next release.


## Prerequisites

* The binary installed Intel® optimizations for TensorFlow 1.15 or 2.1 are preferred. The Intel® optimizations for 
TensorFlow 2.0 is also supported for evaluation. 

  ```bash
  $ pip install intel-tensorflow==1.15.2
  $ pip install intel-quantization
  ```
* The source release repository of Model Zoo for Intel® Architecture is required, if want to execute the quantization
of specific models in Model Zoo as examples.

  ```bash
  $ cd ~
  $ git clone https://github.com/IntelAI/models.git models && cd models
  $ git checkout v1.5.0
  ```

* The source release repository of Intel® AI Quantization Tools for TensorFlow.

  ```bash
  $ cd ~
  $ git clone https://github.com/IntelAI/tools.git quantization && cd quantization
  ```

## Step-by-step Procedure for ResNet-50 Quantization

In this section, the frozen pre-trained model and ImageNet dataset will be required for fully automatic quantization. 

```bash
$ cd ~/quantization/api/models/resnet50
$ wget https://storage.googleapis.com/intel-optimized-tensorflow/models/resnet50_fp32_pretrained_model.pb
```
The TensorFlow models repository provides [scripts and instructions](https://github.com/tensorflow/models/tree/master/research/slim#an-automated-script-for-processing-imagenet-data) to download, process, and convert the ImageNet dataset to the TFRecord format.


1. Run demo script

There are three methods to run the quantization for specific models under `api/examples/`, including bash command for model zoo, bash command for custom model,
and python programming APIs direct call. 

To quantize the models in Model Zoo for Intel® Architecture, the bash commands for model zoo is an easy method with few input parameters.  
```bash
$ cd ~/quantization
$ python api/examples/quantize_model_zoo.py \
--model resnet50 \
--in_graph path/to/resnet50_fp32_pretrained_model.pb \
--out_graph path/to/output.pb \
--data_location path/to/imagenet \
--models_zoo_location ~/models
```

Check the input parameters of pre-trained model, dataset path to match with your local environment. 
And then execute the python script, you will get the fully automatic quantization conversion from FP32 to INT8.


For any custom models that are not supported by Model Zoo for Intel® Architecture, the other bash command `api/examples/quantize_cmd.py` is provided.
The main difference with model zoo bash command is that user needs to prepare the inference command and pass the string as parameter of callback. And then
the callback function will execute the temporary INT8 .pb generated in the middle process to output the min and max log information. Therefore, remove 
the --input_graph parameters and value from the command string for callback function.
   
```bash
$ python api/examples/quantize_cmd.py \ 
                       --input_graph   path/to/resnet50_fp32_pretrained_model.pb \
                       --output_graph  path/to/output.pb \
                       --callback      inference_command_with_small_subset_for_ min_max_log
                       --inputs 'input'
                       --outputs 'predict'
                       --per_channel False
                       --excluded_ops ''
                       --excluded_nodes ''
```
  `--callback`:The command is to execute the inference with small subset of the training dataset to get the min and max log output.
  `--inputs`:The input op names of the graph.
  `--outputs`:The output op names of the grap.
  `--per_channel`:Enable per-channel or not. The per-channel quantization has a different scale and offset for each convolutional kernel.
  `--excluded_ops`:The ops list that excluded from quantization.
  `--excluded_nodes`:The nodes list that excluded from quantization.

The third alternative method to execute the quantization by Python Programming APIs is by Python script directly. 
A template is provided in api/examples/quantize_python.py. The key code is below. 

```python
import os
import intel_quantization.graph_converter as converter

def rn50_callback_cmds():
    # This command is to execute the inference with small subset of the training dataset, and get the min and max log output.

if __name__ == '__main__':
    rn50 = converter.GraphConverter('path/to/resnet50_fp32_pretrained_model.pb', None, ['input'], ['predict'])
    # pass an inference script to `gen_calib_data_cmds` to generate calibration data.
    rn50.gen_calib_data_cmds = rn50_callback_cmds()
    # use "debug" option to save temp graph files, default False.
    rn50.debug = True
    rn50.covert()
```
A [Summarize graph](#summarize-graph) python tool is provided to detect the possible inputs and outputs nodes list of the input .pb graph.

2. Performance Evaluation

Finally, verify the quantized model performance:
 * Run inference using the final quantized graph and calculate the accuracy.
 * Typically, the accuracy target is the optimized FP32 model accuracy values.
 * The quantized INT8 graph accuracy should not drop more than ~0.5-1%.


Check [Intelai/models](https://github.com/IntelAI/models) repository and [ResNet50 README](https://github.com/IntelAI/models/tree/master/benchmarks/image_recognition/tensorflow/resnet50) for TensorFlow models inference benchmarks with different precisions.


## Integration with Model Zoo for Intel Architecture

An integration component with Model Zoo for Intel®  Architecture is provided, that allows users run following models as reference:

- ResNet-50
- ResNet-50 V1_5
- Faster-RCNN
- Inception_V3
- MobileNet_V1
- ResNet-101
- R-FCN
- SSD-MobileNet_V1
- SSD-ResNet34


The model name, launch inference commands for min/max log generation, and specific model quantization parameters are well defined in JSON configuation file `api/config/models.json`.

Take ResNet-50 as an example.

```
{
  "MODEL_NAME": "resnet50",
  "LAUNCH_BENCHMARK_PARAMS": {
    "LAUNCH_BENCHMARK_SCRIPT": "benchmarks/launch_benchmark.py",
    "LAUNCH_BENCHMARK_CMD": [
      " --model-name=resnet50",
      " --framework=tensorflow",
      " --precision=int8",
      " --mode=inference",
      " --batch-size=100",
      " --accuracy-only"
    ],
    "IN_GRAPH": " --in-graph={}",
    "DATA_LOCATION": " --data-location={}"
  },
  "QUANTIZE_GRAPH_CONVERTER_PARAMS": {
    "INPUT_NODE_LIST": [
      "input"
    ],
    "OUTPUT_NODE_LIST": [
      "predict"
    ],
    "EXCLUDED_OPS_LIST": [],
    "EXCLUDED_NODE_LIST": [],
    "PER_CHANNEL_FLAG": false
  }
}
```

- MODEL_NAME: The model name.

- LAUNCH_BENCHMARK_PARAMS
  - LAUNCH_BENCHMARK_SCRIPT: The relative path of running script in Model Zoo.
  - LAUNCH_BENCHMARK_CMD: The parameters to launch int8 accuracy script in Model Zoo.
  - IN_GRAPH: The path of input graph file.
  - DATA_LOCATION: The path of dataset.
  - MODEL_SOURCE_DIR: The path of tensorflow-models.(optional)
  - DIRECT_PASS_PARAMS_TO_MODEL: The parameters directly passed to the model.(optional)

- QUANTIZE_GRAPH_CONVERTER_PARAMS
  - INPUT_NODE_LIST: The input nodes name list of the model. You can use [Summarize graph](#summarize-graph) to get the inputs and outputs of the graph.
  - OUTPUT_NODE_LIST: The output nodes name list of the model.
  - EXCLUDED_OPS_LIST: The list of operations to be excluded from quantization.
  - EXCLUDED_NODE_LIST: The list of nodes to be excluded from quantization.
  - PER_CHANNEL_FLAG: If set True, enables weight quantization channel-wise.



## Tools

### Summarize graph

In order to remove the TensorFlow source build dependency, the independent Summarize graph tool `api/tools/summarize_graph.py` is provided to dump the possible inputs nodes and outputs nodes of the graph. It could be taken as the reference list for INPUT_NODE_LIST and OUTPUT_NODE_LIST parameters
of graph_converter class. 

- If use graph in binary,

```bash
$ python summarize_graph.py --in_graph=path/to/graph --input_binary
```

- Or use graph in text,

```bash
$ python summarize_graph.py --in_graph=path/to/graph
```


## Docker support 

* [Docker]( https://docs.docker.com/install/ ) - Latest version.

* Build a docker layer which contains Inteli® Optimizations for TensorFlow and Intel® AI Quantization Tools for Tensorflow with the command below. 

  ```bash
  $ cd ~/quantization/api/docker
  $ docker build \
       --build-arg HTTP_PROXY=${HTTP_PROXY} \
       --build-arg HTTPS_PROXY=${HTTPS_PROXY} \
       --build-arg http_proxy=${http_proxy} \
       --build-arg https_proxy=${https_proxy} \
       -t quantization:latest -f Dockerfile .
  ```

* Launch quantization script `launch_quantization.py` by providing args as below, this will get user into container environment (`/workspace`) with quantization tools.

  `--docker-image`: Docker image tag from above step (`quantization:latest`).  
  `--in_graph`: Path to your pre-trained model file, which will be mounted inside the container at `/workspace/pretrained_model`.   
  `--out_graph`: When working in the container, all outputs should be saved to `/workspace/output`, so that results are written back to the local machine.  
  `--debug`:Mount the volume and lauch the docker environment to Bash shell environment for debug purpose.   
  `--model_name` and `--models_zoo` are the specific parameters for Model Zoo for Intel® Architecture. If user only want to launch the quantization environment in docker and execute own defined models with `--debug` parameter, both can be skipped. 

* Take the ResNet50 of Model Zoo as an example. 

  ```bash
  $ cd ~/quantization/api
  $ python launch_quantization.py  \
  --docker-image quantization:latest \
  --in_graph=/path/to/in_graph.pb \
  --model_name=resnet50 \
  --models_zoo=/path/to/models_zoo \
  --out_graph=/path/to/output.pb \
  --data_location=/path/to/dataset
  ```

## FAQ

* What's the difference with between Quantization Programming APIs and Tensorflow native quantization?
  
  The Quantization Programming APIs are specified for Intel® Optimizations for TensorFlow based on the MKLDNN enabled build. This APIs call the Tensorflow Python models as extension,
  and provide some special fusion rules, such as, fold_convolutionwithbias_mul, fold_subdivmul_batch_norms, fuse_quantized_conv_and_requantize, mkl_fuse_pad_and_conv,
  rerange_quantized_concat etc. 

* How to build the development environment?
  
  For any code contributers, the .whl is easy to be rebuilt to include the specific code for debugging purpose. Refer the build command below.  

  ```bash
  $ cd ~/quantization/api/
  $ python setup.py bdist_wheel
  $ pip install
  $ pip install dist/*.whl
  ```
  


