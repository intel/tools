#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: EPL-2.0
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import signal
import subprocess
import sys
from argparse import ArgumentParser


class LaunchQuantization(object):
    """Launches quantization job based on the specified args """

    def main(self):
        args, unknown = self.parse_args(sys.argv[1:])
        try:
            self.validate_args(args)
        except (IOError, ValueError) as e:
            print("\nError: {}".format(e))
            sys.exit(1)
        self.run_docker_container(args)

    def parse_args(self, args):

        arg_parser = ArgumentParser(
            add_help=True,
            description="Parse args for quantization interface")

        arg_parser.add_argument(
            "-p", "--pre-trained-model-dir",
            help="Specify the pre-trained models source directory from your local machine",
            dest="pre_trained_model_dir", default=None, required=True)

        arg_parser.add_argument(
            "-i", "--docker-image",
            help="Specify the docker image/tag to use",
            dest="docker_image", default=None, required=True)

        arg_parser.add_argument(
            "-v", "--verbose", help="Print verbose information.",
            dest="verbose", action="store_true")

        arg_parser.add_argument(
            "-t", "--test",
            help="Runs integration tests for quantization tools",
            dest="test", action="store_true")

        return arg_parser.parse_known_args(args)

    def check_for_link(self, arg_name, path):
        """
        Throws an error if the specified path is a link. os.islink returns
        True for sym links.  For files, we also look at the number of links in
        os.stat() to determine if it's a hard link.
        """
        if os.path.islink(path) or \
                (os.path.isfile(path) and os.stat(path).st_nlink > 1):
            raise ValueError("The {} cannot be a link.".format(arg_name))

    def validate_args(self, args):
        """validate the args"""

        # check pre-trained model source directory exists
        pre_trained_model_dir = args.pre_trained_model_dir
        if pre_trained_model_dir is not None:
            if not os.path.exists(pre_trained_model_dir) or \
                    not os.path.isdir(pre_trained_model_dir):
                raise IOError("The pre-trained model source directory {} "
                              "does not exist or is not a directory.".
                              format(pre_trained_model_dir))
            self.check_for_link("pre-trained model source directory", pre_trained_model_dir)

        # Check for spaces in docker image
        if ' ' in args.docker_image:
            raise ValueError("docker image string "
                             "should not have whitespace(s)")

    def run_docker_container(self, args):
        """
        Runs a docker container with the specified image and environment
        variables to start running the quantization job.
        """

        workspace = "/workspace"
        tf_workspace = workspace + "/tensorflow"

        if args.test:
            mount_output = workspace + "/output"
            mount_test_workspace = workspace + "/tests"

            # output and test envs
            env_vars = ["--env", "{}={}".format("MOUNT_OUTPUT", mount_output),
                        "--env", "{}={}".format("TEST_WORKSPACE", mount_test_workspace)]

            # output and test volumes
            test_workspace = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "tests")
            volume_mounts = ["--volume", "{}:{}".format(test_workspace, mount_test_workspace),
                             "--volume", "{}:{}".format(args.pre_trained_model_dir,
                                                        mount_output)]
        else:
            mount_quantization = workspace + "/quantization"

            env_vars = ["--env", "{}={}".format("MOUNT_QUANTIZATION", mount_quantization)]

            volume_mounts = ["--volume", "{}:{}".format(args.pre_trained_model_dir,
                                                        mount_quantization)]

        env_vars += ["--env", "{}={}".format("PRE_TRAINED_MODEL_DIR", args.pre_trained_model_dir),
                     "--env", "{}={}".format("WORKSPACE", workspace),
                     "--env", "{}={}".format("TF_WORKSPACE", tf_workspace)]

        # Add proxy to env variables if any set on host
        for environment_proxy_setting in [
            "http_proxy",
            "ftp_proxy",
            "https_proxy",
            "no_proxy",
        ]:
            if not os.environ.get(environment_proxy_setting):
                continue
            env_vars.append("--env")
            env_vars.append("{}={}".format(
                environment_proxy_setting,
                os.environ.get(environment_proxy_setting)
            ))

        docker_run_cmd = ["docker", "run", "-it"]

        docker_run_cmd = docker_run_cmd + env_vars + volume_mounts + [
            "--privileged", "-u", "root:root", "-w", tf_workspace,
            args.docker_image, "/bin/bash"]

        if args.test:
            docker_run_cmd.append(workspace + "/tests/test_quantization.sh")

        if args.verbose:
            print("Docker run command:\n{}".format(docker_run_cmd))

        self._run_docker_cmd(docker_run_cmd)

    def _run_docker_cmd(self, docker_run_cmd):
        """runs docker proc and exits on ctrl c"""
        p = subprocess.Popen(docker_run_cmd, preexec_fn=os.setsid)
        try:
            p.communicate()
        except KeyboardInterrupt:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)


if __name__ == "__main__":
    util = LaunchQuantization()
    util.main()
