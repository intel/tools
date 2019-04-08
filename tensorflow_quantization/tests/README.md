# Quantization Test Scripts

The `tests` directory contains the scripts and tools that tests the quantization process for tensorflow models.

The `launch_test.sh` script sets up the integration test environment and starts the the quantization process test script `test_quantization.sh`.
To launch the test script:
```
# Clone the quantization tools branch
$ git clone https://github.com/IntelAI/tools.git
$ cd tools/tensorflow_quantization
$ make integration_test
``` 

`test_quantization.sh` tests models such as `ResNet50, ResNet101 and SSD-mobilenet`.
As per model function, the `FP32` pre-trained model is downloaded,
then it uses the defined model parameters to execute and test the quantization steps for the given model.

The `calibration_data` directory contains the `min_max_log.txt` files per model, which is used in the quantization process to
freeze the `dynamic range quantized graph`.

There are also unit and lint tests for both python and c++.
To launch these tests:
```
$ make cpplint
$ make lint
$ make unit_test
$ make cpp_test
```
There are other `make` commands that can be found in the `Makefile` for running specific tests on certain python versions.
Note that `cpp_test` expects to be ran within an already-built tensorflow repo, and `make cpplint` will only work on Linux.
