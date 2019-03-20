# Quantization Test Scripts

The `tests` directory contains the scripts and tools that tests the quantization process for tensorflow models.

The `launch_test.sh` script sets up the test environment and starts the the quantization process test script `test_quantization.sh`.
To launch the test script:
```
# Clone the quantization tools branch
$ git clone https://github.com/NervanaSystems/tools.git
$ cd tools/tensorflow-quantization/tests
$ bash launch_test.sh
``` 

`test_quantization.sh` tests models such as `ResNet50, ResNet101 and SSD-mobilenet`.
As per model function, the `FP32` pre-trained model is downloaded,
then it uses the defined model parameters to execute and test the quantization steps for the given model.

The `calibration_data` directory contains the `min_max_log.txt` files per model, which is used in the quantization process to
freeze the `dynamic range quantized graph`.