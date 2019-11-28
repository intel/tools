# Quantization Python Programming API Examples

Content:
* [Goal](#goal)
* [Prerequisites](#prerequisites)
* [Step-by-step Procedure for ResNet50 Quantization](#step-by-step-procedure-for-resnet50-quantization)
* [More verified models](#more-verified-models)
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
```bash
$ cd ~
$ git clone https://github.com/IntelAI/tools.git quantization && cd quantization
$ export PYTHONPATH=${PYTHONPATH}:${PWD}
```



## Step-by-step Procedure for ResNet50 Quantization

In this section, the frozen pre-trained model and ImageNet dataset will be required for fully automatic quantization. 

```bash
$ cd ~/quantization/api/models/resnet50
$ wget https://storage.googleapis.com/intel-optimized-tensorflow/models/resnet50_fp32_pretrained_model.pb
```

If want to enable the example of ResNet50 v1.5, please download the frozen pre-trained model from the link below.

```bash
$ cd ~/quantization/api/models/resnet50v1_5
$ wget https://zenodo.org/record/2535873/files/resnet50_v1.pb
```

The TensorFlow models repo provides [scripts and instructions](https://github.com/tensorflow/models/tree/master/research/slim#an-automated-script-for-processing-imagenet-data) to download, process, and convert the ImageNet dataset to the TF records format.

1. Download TensorFlow source, patch Graph Transform Tool and install the TensorFlow.
```bash
$ cd ~/
$ git clone https://github.com/tensorflow/tensorflow.git
$ cd tensorflow
$ git checkout v1.14.0
$ cp ../quantization/tensorflow_quantization/graph_transforms/*  tensorflow/tools/graph_transforms/
```
And then [build and install TensorFlow from Source with Intel® MKL](https://software.intel.com/en-us/articles/intel-optimization-for-tensorflow-installation-guide).



2. Run demo script
```bash
$ python api/quantize_model.py \
--model=resnet50 \
--model_location=path/to/resnet50_fp32_pretrained_model.pb \
--data_location=path/to/imagenet
```

Check the input parameters of pre-trained model, dataset path to match with your local environment. And then execute the python script, you will get the fully automatic quantization conversion from FP32 to INT8.



3. Performance Evaluation

Finally, verify the quantized model performance:
 * Run inference using the final quantized graph and calculate the model accuracy.
 * Typically, the accuracy target is the optimized FP32 model accuracy values.
 * The quantized `INT8` graph accuracy should not drop more than ~0.5-1%.

 Check [Intelai/models](https://github.com/IntelAI/models) repository and [ResNet50](https://github.com/IntelAI/models/tree/master/benchmarks/image_recognition/tensorflow/resnet50) README for TensorFlow models inference benchmarks with different precisions.



## More verified models

The following models are also verified:

- [SSD-MobileNet](#ssd-mobilenet)
- [SSD-ResNet34](#ssd-resnet34)



### SSD-MobileNet

Download and extract the pre-trained SSD-MobileNet model from the [TensorFlow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md#coco-trained-models). The downloaded .tar file includes a `frozen_inference_graph.pb` which will be used as the input graph for quantization.

```bash
$ wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz
$ tar -xvf ssd_mobilenet_v1_coco_2018_01_28.tar.gz
```



Follow the [instructions](https://github.com/IntelAI/models/blob/master/benchmarks/object_detection/tensorflow/ssd-mobilenet/README.md#int8-inference-instructions) to prepare your local environment and build ssd_mobilenet_callback_cmds() command to generate the min. and max. ranges for the model calibration.

```python
_INPUTS = ['image_tensor']
_OUTPUTS = ['detection_boxes', 'detection_scores', 'num_detections', 'detection_classes']


def ssd_mobilenet_callback_cmds():
    # This command is to execute the inference with small subset of the training dataset, and get the min and max log output.

if __name__ == '__main__':
    c = convert.GraphConverter('path/to/ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb', None, _INPUTS, _OUTPUTS, excluded_ops=['ConcatV2'], per_channel=True)
    c.gen_calib_data_cmds = ssd_mobilenet_callback_cmds()
    c.convert()
```





### SSD-ResNet34

Download the pretrained model:

```bash
$ wget https://storage.googleapis.com/intel-optimized-tensorflow/models/ssd_resnet34_fp32_bs1_pretrained_model.pb
```



Follow the [instructions](https://github.com/IntelAI/models/blob/master/benchmarks/object_detection/tensorflow/ssd-resnet34/README.md#int8-inference-instructions) to prepare your int8 accuracy commands to generate the min. and max. ranges for the model calibration.

```python
_INPUTS = ['input']
_OUTPUTS = ['v/stack', 'v/Softmax']


def ssd_resnet34_callback_cmds():
    # This command is to execute the inference with small subset of the training dataset, and get the min and max log output.


if __name__ == '__main__':
    c = convert.GraphConverter('path/to/ssd_resnet34_fp32_bs1_pretrained_model.pb', None, _INPUTS, _OUTPUTS, excluded_ops=['ConcatV2'])
    c.gen_calib_data_cmds = ssd_resnet34_callback_cmds()
    c.convert()
```





## Docker support

* For docker environment, the procedure is same as above. 

