#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2020 Intel Corporation
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import signal
import subprocess
import sys
from argparse import ArgumentParser
from argparse import ArgumentTypeError


class LaunchQuantization():
    """Launches quantization job based on the specified args """
    def main(self):
        try:
            args, _ = self.parse_args(sys.argv[1:])
        except ArgumentTypeError as e:
            print("\nError: {}".format(e))
            sys.exit(1)
        self.run_docker_container(args)

    def parse_args(self, args):

        arg_parser = ArgumentParser(
            add_help=True,
            description="Parse args for quantization interface")

        arg_parser.add_argument(
            "-i", "--docker-image",
            help="Specify the docker image/tag to use",
            dest="docker_image", default=None, required=True, type=self.check_no_spaces)

        arg_parser.add_argument(
            "-g", "--in_graph",
            help="Specific the path of in_graph",
            dest="in_graph", default=None, type=self.check_valid_filename)

        arg_parser.add_argument(
            "-o", "--out_graph",
            help="Specific the path of out_graph",
            dest="out_graph", default=None, type=self.check_valid_in_dir)

        arg_parser.add_argument(
            "-d", "--data_location",
            help="Specific the path of dataset",
            dest="data_location", default=None, type=self.check_valid_file_or_dir)

        arg_parser.add_argument(
            "-z", "--models_zoo",
            help="Specific the path of models zoo",
            dest="models_zoo", default=None, type=self.check_valid_folder)

        arg_parser.add_argument(
            "-n", "--model_name",
            help="Specific the model name",
            dest="model_name", default="resnet50")

        arg_parser.add_argument(
            "-s", "--models_source_dir",
            help="Specific the path of model source",
            dest="models_source_dir", default=None, type=self.check_valid_folder)

        arg_parser.add_argument(
            "--debug",
            help="Launch debug mode which does not execute scrpit when \
                  when running in a docker container",
            action="store_true")

        arg_parser.add_argument(
            "--intermediate",
            help="Generate intermediate graph",
            action="store_true")

        return arg_parser.parse_known_args(args)

    def check_for_link(self, value):
        """
        Throws an error if the specified path is a link. os.islink returns
        True for sym links.  For files, we also look at the number of links in
        os.stat() to determine if it's a hard link.
        """
        if os.path.islink(value) or \
                (os.path.isfile(value) and os.stat(value).st_nlink > 1):
            raise ArgumentTypeError("{} cannot be a link.".format(value))

    def check_no_spaces(self, value):
        """checks for spaces in string"""
        if ' ' in value:
            raise ArgumentTypeError("{} should not have whitespace(s).")
        return value

    def check_valid_filename(self, value):
        """verifies filename exists and isn't a link"""
        if value is not None:
            if not os.path.isfile(value):
                raise ArgumentTypeError("{} does not exist or is not a file.".
                                        format(value))
            self.check_for_link(value)
        return value

    def check_valid_folder(self, value):
        """verifies filename exists and isn't a link"""
        if value is not None:
            if not os.path.isdir(value):
                raise ArgumentTypeError("{} does not exist or is not a directory.".
                                        format(value))
            self.check_for_link(value)
        return value

    def check_valid_file_or_dir(self, value):
        """verfies file/dir exists and isn't a link"""
        if value is not None:
            if not os.path.exists(value):
                raise ArgumentTypeError("{} does not exist.".format(value))
            self.check_for_link(value)
        return value

    def check_valid_in_dir(self, value):
        """
        verifies is the dirname of the file is exist and the file is not a link
        """
        if value is not None:
            dir_ = os.path.dirname(value)
            if not os.path.isdir(dir_):
                raise ArgumentTypeError("{} does not exist.".format(dir_))
            if os.path.islink(value):
                raise ArgumentTypeError("{} cannot be a link.".format(value))
        return value

    def run_docker_container(self, args):
        """
        Runs a docker container with the specified image and environment
        variables to start running the quantization job.
        """
        workspace = "/workspace"
        data_location = workspace + "/dataset"
        quantization = workspace + "/quantization"
        in_graph = workspace + "/pretrained_models/in_graph.pb"

        out_graph_dir = workspace + "/output/"
        out_graph = out_graph_dir + os.path.basename(args.out_graph)
        args.out_graph = os.path.dirname(args.out_graph)

        models_zoo = None if args.models_zoo is None else workspace + "/models_zoo"
        models_source_dir = None if args.models_source_dir is None else workspace + "/models_source_dir"
        REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                 os.path.pardir,
                                                 os.path.pardir))

        start_cmd = quantization + "/api/docker/start.sh"

        env_vars = ["--env", "{}={}".format("WORKSPACE", workspace),
                    "--env", "{}={}".format("IN_GRAPH", in_graph),
                    "--env", "{}={}".format("OUT_GRAPH", out_graph),
                    "--env", "{}={}".format("DATA_LOCATION", data_location),
                    "--env", "{}={}".format("MODELS_ZOO", models_zoo),
                    "--env", "{}={}".format("MODELS_SOURCE_DIR", models_source_dir),
                    "--env", "{}={}".format("DEBUG", args.intermediate),
                    "--env", "{}={}".format("MODEL_NAME", args.model_name)]

        volume_mounts = []

        if args.models_source_dir is not None:
            volume_mounts += ["--volume", "{}:{}".format(args.models_source_dir, models_source_dir)]
        if args.models_zoo is not None:
            volume_mounts += ["--volume", "{}:{}".format(args.models_zoo, models_zoo)]

        volume_mounts += ["--volume", "{}:{}".format(REPO_PATH, quantization),
                          "--volume", "{}:{}".format(args.in_graph, in_graph),
                          "--volume", "{}:{}".format(args.out_graph, out_graph_dir),
                          "--volume", "{}:{}".format(args.data_location, data_location)]

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
            "--privileged", "-u", "root:root", "-w", workspace,
            args.docker_image, "/bin/bash"]

        if not args.debug:
            del docker_run_cmd[2]
            docker_run_cmd.append(start_cmd)

        print("Docker run command:\n{}".format(docker_run_cmd))

        self._run_docker_cmd(docker_run_cmd)

    def _run_docker_cmd(self, docker_run_cmd):
        """runs docker proc and exits on ctrl c"""
        p = subprocess.Popen(docker_run_cmd, preexec_fn=os.setsid)
        try:
            _, err = p.communicate()
        except KeyboardInterrupt:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)

        if p.returncode != 0 and p.returncode != 124:
            raise SystemExit(
                "\nERROR running the following docker command:\n{}\nDocker error code: {}\nstderr: {}".format(
                    " ".join(docker_run_cmd), p.returncode, err))


if __name__ == "__main__":
    util = LaunchQuantization()
    util.main()
