# Quantization Test Scripts

The `tests` directory includes test scripts for `lint` , `unit` and `integration` tests.


### Lint Tests
To run `lint` tests for `python` and `c++` use below make commands
```
$ make lint
$ make cpplint
```


### Unit Tests
To run `unit` tests for `python` and `c++` use below make commands
```
$ make unit_test
$ make cpp_test
```

### Integration tests
* The `integration` folder contains scripts to launch integration tests for pre-defined models inside `test_quantization.sh` script.
* The `launch_test.sh` script sets up the integration test environment and starts `test_quantization.sh` to initiate quantization process.
* As per model function, the `FP32` pre-trained model is downloaded and uses defined transformations steps to test the quantization process for given model.
* The `calibration_data` directory contains the `min_max_log.txt` files per model, which is used in the quantization process to
freeze the `dynamic range quantized graph`.

To launch integration suite, follow as below:
```
# Clone the quantization tools branch
$ git clone https://github.com/IntelAI/tools.git
$ cd tools/tensorflow_quantization
$ make integration_test
```

>NOTE:
> * There are other `make` commands that can be found in the `Makefile` for running specific tests on certain python versions.
> * `cpp_test` expects to be ran within an already-built tensorflow repo
> * `make cpplint` will only work on Linux.
