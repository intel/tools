# Quantization Examples 
We will give two examples of how to use quantization tools with intel tensorflow 1.15.x, 2.0.x and 2.1.x

## Contents
1. quantize intel model zoo models  
2. quantize official classification models  

### Quantize Intel model zoo Models  
<b>Model List:</b> 'resnet50', 'resnet50v1_5', 'resnet101', 'ssd-resnet34', 'mobilenet_v1','inceptionv3'
#### step by step  
1. environment prepare  
Please install intel tensorflow and download paired intel model zoo version.  
For example: `pip intall intel-tensorflow==1.15.2`  
Intel model zoo : https://github.com/IntelAI/models  
TF 1.15.x pair with intel model zoo v1.5.0  
TF 2.0.x & TF 2.1.x pair with intel model zoo v1.6.0  
2. prepare dataset and pbs  
Please check readme for specific model in `models/benchmarks/README.md`
3. quantize command  
```
$ export PYTHONPATH=${PYTHONPATH}:PATH_OF_TOOLS/api
$ python quantize_model_zoo.py \
       --model mobilenet_v1 \
       --in_graph /tf_dataset/pre-trained-models/mobilenet_v1/fp32/mobilenet_v1_1.0_224_frozen.pb \
       --out_graph ${WORKSPACE}/mobilenet_v1-quantize-${HOSTNAME}.pb \
       --data_location /tf_dataset/dataset/TF_Imagenet_FullData \
       --models_zoo_location ${WORKSPACE}/models/
```
If the model need `--model-source-dir` please download: https://github.com/tensorflow/models and checkout the required branch mentioned in intel model zoo readme.

### Quantize Official Classification Models
<b>Model List for TF1.15.x and TF2.0.x:</b> 'inception_v1','inception_v4','vgg_16','vgg_19','mobilenet_v1','resnet_v1_152','resnet_v1_50'  
<b>Model List for TF2.1.x:</b> 'inception_v1', 'inception_v2', 'inception_v4','vgg_16', 'vgg_19', 'mobilenet_v2', 'mobilenet_v1', 'resnet_v1_152', 'resnet_v1_50'  

#### step by step
1. environment prepare  
Please install intel tensorflow 1.15.x or 2.0.x or 2.1.x  
2. prepare dataset and pbs  
Get checkpoint from `https://github.com/tensorflow/models/blob/master/research/slim/README.md` and generate frozen pb for quantize.   
3. quantize command  
```
$ export PYTHONPATH=${PYTHONPATH}:PATH_OF_TOOLS/api
python quantize_model_oob_slim.py \
        --model ${model_name} \
        --model_location /tf_dataset/pre-train-model-slim/pbfile/frozen_pb/frozen_${model_name}.pb \
        --data_location /tf_dataset/dataset/TF_Imagenet_FullData \
        --out_graph ${WORKSPACE}/${model_name}_quantization.pb 
```