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



## Goal

The Quantization Python programming API is to:
* Unify the quantization tools calling entry, 
* Remove the Tensorflow source build dependency,
* Transparent the model quantization process, 
* Reduce the quantization steps,
* Seamlessly adapt to inference with Python script.


This feature is under active development, and more intelligent features will come in next release.



## Prerequisites

* The binary installed Intel® optimizations for TensorFlow 1.14 or 1.15 are preferred. The Intel® optimizations for 
TensorFlow 2.0 is also supported for evaluation. 

  ```bash
  $ pip install intel-tensorflow==1.15.0
  ```
* The source release repository of Model Zoo for Intel® Architecture, if want to execute the quantization
of specific models in Model Zoo as examples.

  ```bash
  $ cd ~
  $ git clone https://gitlab.devtools.intel.com/intelai/models.git models && cd models
  $ git checkout develop-tf-1.15
  ```

* The source release repository of Intel® AI Quantization Tools for TensorFlow.

  ```bash
  $ cd ~
  $ git clone https://gitlab.devtools.intel.com/intelai/tools.git  quantization && cd quantization
  $ git checkout r1.0_alpha2_rc
  $ pip install api/bin/intel_quantization-1.0a0-py3-none-any.whl
  ```

## Step-by-step Procedure for ResNet-50 Quantization

In this section, the frozen pre-trained model and ImageNet dataset will be required for fully automatic quantization. 

```bash
$ cd ~/quantization/api/models/resnet50
$ wget https://storage.googleapis.com/intel-optimized-tensorflow/models/resnet50_fp32_pretrained_model.pb
```
The TensorFlow models repository provides [scripts and instructions](https://github.com/tensorflow/models/tree/master/research/slim#an-automated-script-for-processing-imagenet-data) to download, process, and convert the ImageNet dataset to the TFRecord format.


1. Run demo script
```bash
$ cd ~/quantization
$ python api/quantize_model.py \
--model resnet50 \
--in_graph path/to/resnet50_fp32_pretrained_model.pb \
--out_graph path/to/output.pb \
--data_location path/to/imagenet \
--models_zoo_location ~/models
```

Check the input parameters of pre-trained model, dataset path to match with your local environment. 
And then execute the python script, you will get the fully automatic quantization conversion from FP32 to INT8.


The other alternative method to execute the quantization by Python Programming APIs is by Python script directly. A [Summarize graph](#summarize-graph) python tool is provided to detect the possible inputs and outputs nodes list of the input .pb graph.

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

* For docker environment, the procedure is same as above. 

