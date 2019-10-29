# Quantization Python Programming API Example with ResNet-50

Content:
* [Goal](#goal)
* [Prerequisites](#prerequisites)
* [Step-by-step Procedure for ResNet-50 Quantization](#step-by-step-procedure-for-resnet-50-quantization)
* [Docker support](#docker-support)



## Goal

The Quantization Python programming API is to:
* Unify the quantization tools calling entry, 
* Transparent the model quantization process, 
* Reduce the quantization steps,
* Seamlessly adpat to inference with python script.

This feature is under active development, and more intelligent features will come in next release.



## Prerequisites

* TensorFlow build and install from source knowledge are required, as the Quantization Python Programming API extends the transform functions of [Graph Transform Tool](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/graph_transforms/README.md) in TensorFlow.
* The source release repo of Intel® AI Quantization Tools for TensorFlow.
```
$ cd ~
$ git clone https://github.com/IntelAI/tools.git quantization && cd quantization
$ export PYTHONPATH=${PYTHONPATH}:${PWD}
```



## Step-by-step Procedure for ResNet-50 Quantization

In this section, the frozen pre-trained model and ImageNet dataset will be required for fully automatic quantization. 

```
$ cd ~/quantization/api/models/resnet50
$ wget https://storage.googleapis.com/intel-optimized-tensorflow/models/resnet50_fp32_pretrained_model.pb
```

If want to enable the example of ResNet-50 v1.5, please download the frozen pre-trained model from the link below.

```
$ cd ~/quantization/api/models/resnet50v1_5
$ wget https://zenodo.org/record/2535873/files/resnet50_v1.pb
```

The TensorFlow models repo provides [scripts and instructions](https://github.com/tensorflow/models/tree/master/research/slim#an-automated-script-for-processing-imagenet-data) to download, process, and convert the ImageNet dataset to the TF records format.
We assume the imagenet dataset is on `~/quantization/api/models/imagenet` directory. 

1. Download TensorFlow source, patch Graph Transform Tool and install the TensorFlow.
```
$ cd ~/
$ git clone -b r1.14 --single-branch https://github.com/tensorflow/tensorflow.git
$ cd tensorflow
$ cp ../quantization/tensorflow_quantization/graph_transforms/*  tensorflow/tools/graph_transforms/
```
And then [build and install TensorFlow from Source with Intel® MKL](https://software.intel.com/en-us/articles/intel-optimization-for-tensorflow-installation-guide).

2. Modify the input parameters and launch the Python programming API entry.
```
$ cd ~/quantization/api
$ vim demo.py
```
And you will see the quantization calling code as below. The `rn50_callback_cmds()` is a callback function
to tell the graph_converter class how to execute the inference with dataset in order to generate 
the min_max_log file for freezing requantization ranges.

```
import os
import graph_converter as converter # You may need to change the path depend on your own script path.

_RN50_MODEL = os.path.join(os.environ['HOME'], 'quantization/api/models/resnet50/resnet50_fp32_pretrained_model.pb')
_DATA_LOC = os.path.join(os.environ['HOME'], 'quantization/api/models/imagenet')

def rn50_callback_cmds():
    script = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models/resnet50/accuracy.py')
    flags = ' --batch_size=50' + \
            ' --num_inter_threads=2' + \
            ' --num_intra_threads=28' + \
            ' --input_graph={}' + \
            ' --data_location={}'.format(_DATA_LOC) + \
            ' --num_batches 10'
    return script + flags

if __name__ == '__main__':
    # ResNet50 v1.0 quantization example
    rn50 = converter.GraphConverter(_RN50_MODEL, None, ['input'], ['predict'])
    rn50.gen_calib_data_cmds = rn50_callback_cmds()
    rn50.convert()
```
Check the input parameters of pre-trained model, dataset path to match with your local environment. And then execute the python script, you will get the fully automatic quantization conversion from FP32 to INT8.
```
$ python demo.py
```
3. Performance Evaluation

Finally, verify the quantized model performance:
 * Run inference using the final quantized graph and calculate the model accuracy.
 * Typically, the accuracy target is the optimized FP32 model accuracy values.
 * The quantized `INT8` graph accuracy should not drop more than ~0.5-1%.

 Check [Intelai/models](https://github.com/IntelAI/models) repository and [ResNet50](https://github.com/IntelAI/models/tree/master/benchmarks/image_recognition/tensorflow/resnet50) README for TensorFlow models inference benchmarks with different precisions.



## Docker support

* For docker environment, the procedure is same as above. 

