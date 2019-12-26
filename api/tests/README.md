# Running Tests

## Prerequisites

 - Python 3.4, 3.5 or 3.6
 - Python developer libraries (`python-dev` for Ubuntu or `python-devel` for CentOS 7)
 - [tox](https://tox.readthedocs.io)
 - [requirements](https://github.com/IntelAI/tools/tree/master/api/tests/requirements.txt)

Change your directory to `tools/api`:
```bash
$ cd tools/api
```

## Running all tests

To run style checks and unit tests:
```bash
$ make test
```

## Running style checks only

To run style checks only:
```bash
$ make lint
```

## Running unit tests only

To run unit tests only:
```bash
$ make unit_test
```
