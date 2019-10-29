# Copyright (c) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#	http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Set PY_VERSION to 2 or 3 to create different envs without using tox
PY_VERSION ?= $(shell python --version 2>&1  | cut -c8)

# Get OS so as to not run cpplint if on a Mac
# TODO: is Windows supported?
OS := $(shell uname)

.PHONY: venv

all: venv

venv:
	@echo "Creating Python${PY_VERSION} virtualenv"
	@tox -e py${PY_VERSION}-venv

venv-all: venv
	@echo "Creating Python2 and 3 virtualenv in .venv and .venv3 dirs"
	@tox -e py2-venv -e py3-venv

tox:
	tox


ifeq ($(OS), Darwin)
cpplint:
	@echo "Not running cpp lint tests on Mac, it will fail"
else
# Runs Google required linting checks
# buildifier doesn't write out to files, it just edits them in place
# so we'll check if a `git diff` shows anything
cpplint:
	@echo "Running lint checks for c++ code. NOTE: Google check can't run on Mac."
	find . -name "*.cc" | xargs -n1 tests/cpplint.py
	find . -name "*.cc" | xargs -I '{}' \
		sh -c 'tests/clang-format -style=Google {} > {}-temp.cc; \
			   echo "File to diff: {}-temp.cc"; \
			   diff {} {}-temp.cc | tee -a diffs.txt; \
			   rm {}-temp.cc'; \
		sh -c "if [ -s diffs.txt ]; then rm diffs.txt && exit 1; else rm diffs.txt; fi"
	find . -name "BUILD" | xargs -I '{}' \
		sh -c 'tests/buildifier {}; \
			   git diff {} | tee -a diffs.txt'; \
		sh -c "if [ -s diffs.txt ]; then rm diffs.txt && exit 1; else rm diffs.txt; fi"
endif

lint: cpplint
	@echo "Running style check..."
	@tox -e py2-lint -e py3-lint

lint2: cpplint
	@echo "Running style check..."
	@tox -e py2-lint

lint3: cpplint
	@echo "Running style check..."
	@tox -e py3-lint

unit_test:
	@echo "Running unit tests..."
	@tox -e py2-unit -e py3-unit

unit_test2:
	@echo "Running unit tests python 2..."
	@tox -e py2-unit

unit_test3:
	@echo "Running unit tests python 3..."
	@tox -e py3-unit

integration_test:
	@echo "Running integration tests..."
	@cd tests/integration && bash launch_test.sh

cpp_test:
	@echo "SHOULD ONLY BE RAN WHERE TENSORFLOW CPP CODE IS COMPILED, IN TENSORFLOW WORKSPACE"
	bazel test tensorflow/tools/graph_transforms:all

test: lint unit_test

test2: lint2 unit_test2

test3: lint3 unit_test3

clean:
	rm -rf .venv .venv3 .tox
