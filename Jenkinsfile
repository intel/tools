node() {
    deleteDir()
    // Create a workspace path.  We need this to be 79 chars max, otherwise some nodes fail.
    // The workspace path varies by node so get that path, and then add on 10 chars of a UUID string.
    ws_path = "$WORKSPACE".substring(0, "$WORKSPACE".indexOf("workspace/") + "workspace/".length()) + UUID.randomUUID().toString().substring(0, 10)
    ws(ws_path) {
        // pull the code
        dir('tools') {
            checkout scm
            withCredentials([string(credentialsId: 'intel-models-repo', variable: 'INTEL_MODELS_REPO')]) {
                checkout([$class                           : 'GitSCM',
                    branches                         : [[name: 'develop']],
                    browser                          : [$class: 'AssemblaWeb', repoUrl: ''],
                    doGenerateSubmoduleConfigurations: false,
                    extensions                       : [[$class           : 'RelativeTargetDirectory',
                                                         relativeTargetDir: 'models']],
                    submoduleCfg                     : [],
                    userRemoteConfigs                : [[credentialsId: 'aipgbot-orca',
                                                         url          : "${INTEL_MODELS_REPO}"]]])
            }
        }

        stage('Install dependencies') {
            sh """
            #!/bin/bash -x
            set -e
            # don't know OS, so trying both apt-get and yum install
            sudo apt-get install -y python3-dev python3-pip || sudo yum install -y python36-devel.x86_64 python-pip python36-pip

            # virtualenv 16.3.0 is broken do not use it
            sudo python2 -m pip install --no-cache-dir --upgrade pip==19.0.3 virtualenv!=16.3.0 tox==3.8.6
            sudo python3 -m pip install --no-cache-dir --upgrade pip==19.0.3 virtualenv!=16.3.0 tox==3.8.6
            """
        }

        stage('Style tests') {
            sh """
            #!/bin/bash -x
            set -e

            cd tools/tensorflow_quantization
            make lint
            """
        }

        stage('Integration tests') {
            try
            {
                sh """
                #!/bin/bash -x
                set -e

                # dataset real paths, to be changed when running the integration tests on a local machine.
                export IMAGENET_TF_DATASET=/tf_dataset/dataset/TF_Imagenet_FullData
                export COCO_TF_DATASET=/tf_dataset/dataset/fastrcnn/coco-data
                export COCO_TF_SSDVGG16=/tf_dataset/dataset/SSDvgg16/val2017_tfrecords

                cd tools/tensorflow_quantization
                export INTEL_MODELS=${WORKSPACE}/tools/models
                sudo -E make integration_test
                """
            } finally {
                sh """
                #!/bin/bash -x
                set -e

                MOUNTED_DIRECTORY=${WORKSPACE}/tools/tensorflow_quantization/mounted_dir
                # clean up after the test is done
                if [ -d \$MOUNTED_DIRECTORY ]; then
                    sudo rm -rf \$MOUNTED_DIRECTORY
                fi
                """
            }
        }

        stage('Unit tests') {
            sh """
            #!/bin/bash -x
            set -e

            sudo docker run --rm -e https_proxy -e http_proxy -e HTTPS_PROXY -e HTTP_PROXY -e no_proxy -e NO_PROXY quantization:latest /bin/bash -c "bazel test --config=mkl tensorflow/tools/quantization:quantize_graph_test"
            """
        }

        stage('C++ tests') {
            sh """
            #!/bin/bash -x
            set -e

            sudo docker run --rm -e https_proxy -e http_proxy -e HTTPS_PROXY -e HTTP_PROXY -e no_proxy -e NO_PROXY quantization:latest /bin/bash -c "bazel test --config=mkl tensorflow/tools/graph_transforms:all"
            """
        }
    }
}
