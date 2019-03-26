# Graph Transforms

TensorFlow's [documentation](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/graph_transforms/README.md#transform-reference)
includes descriptions of the graph transforms in their respository.
The list below includes graph transforms developed by Intel and a
description of their usage.

| Graph Transform                      | Description      |
|--------------------------------------|------------------|
| fold_convolutionwithbias_mul         | This graph transform folds the scalar value of the 'Mul' operation into the filter parameter of Conv2D op and into Bias Op. |
| fold_subdivmul_batch_norms           | This transform removes BatchNorm nodes in order to improve performance. The BatchNorm operation may be present as either a single operation node or natively as subtraction followed by real division followed by multiplication. The BatchNorm (Sub-RealDiv-Mul) operator can be removed using the mean, variance and gamma constant nodes and modifying the weights and beta constant nodes, thus folding it into the convolution and bias neighboring operators. |
| fuse_quantized_conv_and_requantize   | This transform is run after the requantization ranges are frozen. It goes through all the nodes and fuses quantized convolution with requantize op. |
| mkl_fuse_pad_and_conv                | This transform is run after the quantize_graph.py step. It goes through all the nodes in the graph and finds pad &rarr; conv &rarr; bias &rarr; relu pattern and fuses it to QuantizedConv2DWithBiasAndRelu op. |
| rerange_quantized_concat             | This transform makes all inputs to the quantized concat in the same quantization range, if possible. If all inputs are in the same range, TensorFlow quantized concat will take the optimal path. |