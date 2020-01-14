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

.PHONY: venv

all: venv

venv:
	@echo "Creating Python3 virtualenv"
	@tox -e py3-venv

tox:
	tox

lint:
	@echo "Running style checks..."
	tox -e py3-lint

unit_test:
	@echo "Running unit tests..."
	@tox -e py3-unit

test: lint # unit_test

clean:
	rm -rf .venv .tox
