#
#  -*- coding: utf-8 -*-
#
#  Copyright (c) 2019 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import os
import argparse

from tensorflow.python.platform import app

import intel_quantization.graph_converter as converter

def model_callback_cmds(data_location,output_shape,image_size):
    script = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          'include/eval_image_classifier_optimize.py')
    flags = " --batch-size=50" + \
            " --input-graph={}" + \
            " --data-location={}".format(data_location) + \
            " --env=mkl --steps=100 -i input -o {}".format(output_shape) + \
            " --image_size={}".format(image_size) + \
            " --accuracy-only "
    return 'python ' + script + flags

def main(_):
    c = None

    per_channel_value = False
    output_shape = args.model + "/predictions/Reshape_1"
    image_size=224

    if args.model == 'inception_v1':
        output_shape = 'InceptionV1/Logits/Predictions/Reshape_1'
    
    elif args.model == 'inception_v2':
        output_shape = 'InceptionV2/Predictions/Reshape_1'

    elif args.model == 'inception_v4':
        output_shape = 'InceptionV4/Logits/Predictions'
        image_size=299

    elif args.model == 'mobilenet_v1':
        per_channel_value = True
        output_shape = 'MobilenetV1/Predictions/Reshape_1'

    elif args.model == 'mobilenet_v2':
        per_channel_value = True
        output_shape = 'MobilenetV2/Predictions/Reshape_1'

    elif args.model == 'vgg_16':
        output_shape = 'vgg_16/fc8/squeezed'

    elif args.model == 'vgg_19':
        output_shape = 'vgg_19/fc8/squeezed'

    elif args.model == 'nasnet_large' or args.model == 'pnasnet_large':
        output_shape = 'final_layer/predictions'
        image_size=331

    if per_channel_value:
        c = converter.GraphConverter(args.model_location, args.out_graph, ['input'], [output_shape],
                                     per_channel=True)
    else:
        c = converter.GraphConverter(args.model_location, args.out_graph, ['input'], [output_shape])
    
    c.debug = True
    c.gen_calib_data_cmds = model_callback_cmds(args.data_location,output_shape,image_size)
    c.convert()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet50', help='The model name')
    parser.add_argument('--model_location', type=str, default=None, help='The original fp32 frozen graph')
    parser.add_argument('--data_location', type=str, default=None, help='The dataset in tfrecord format')
    parser.add_argument('--out_graph', type=str, default=None, help='The path to generated output int8 frozen graph.')

    args = parser.parse_args()

    app.run()

