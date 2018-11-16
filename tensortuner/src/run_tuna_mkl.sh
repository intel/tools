#!/bin/bash
set -e

if [ $# -ne 3 ];
then
  echo "run_tuna_mkl.sh <model> <batch_size> <cpu>"
  # TODO: what can the other options be?
  echo "<cpu> options: skl"
  exit 0
fi

model=$1
batch_size=$2
cpu=$3

#############################
# Configs

LOG_DIR=${LOG_DIR:-/tmp}
TF_CNN_DIR=${TF_CNN_DIR:-~/private-tensorflow-benchmarks/scripts/tf_cnn_benchmarks}

#############################
# Tuna configs

CFG_SEARCH=nm
#CFG_SEARCH=exhaustive
CFG_INFERENCE=False
CFG_INIT_RADIUS=0.9
CFG_OUTPUT=output

#############################

mkdir -p ${LOG_DIR}/${cpu}/${CFG_OUTPUT}/${CFG_SEARCH}/mkl

# We set this for all TF-CNN benchmarks in JSON.
if [ "$cpu" = "skl" ];
then
  omp_min=14
  omp_max=56
  omp_step=7
  intra_op_min=14
  intra_op_max=56
  intra_op_step=7
  inter_op_min=1
  inter_op_max=4
  inter_op_step=1
else
  omp_min=11
  omp_max=44
  omp_step=11
  intra_op_min=11
  intra_op_max=44
  intra_op_step=11
  inter_op_min=1
  inter_op_max=4
  inter_op_step=1
fi

if [ "$CFG_INFERENCE" = "True" ];
then
  LOG=${LOG_DIR}/${cpu}/${CFG_OUTPUT}/${CFG_SEARCH}/mkl/${model}.inference.tuna.${CFG_SEARCH}.log
else
  LOG=${LOG_DIR}/${cpu}/${CFG_OUTPUT}/${CFG_SEARCH}/mkl/${model}.train.tuna.${CFG_SEARCH}.log
fi

INIT_RADIUS=${CFG_INIT_RADIUS} \
${HARMONY_HOME}/bin/tuna \
STRATEGY=${CFG_SEARCH}.so LAYERS=log.so \
LOG_FILE=${LOG} \
-q -v -n=200 \
-i=interop,${inter_op_min},${inter_op_max},${inter_op_step} \
-i=intraop,${intra_op_min},${intra_op_max},${intra_op_step} \
-i=omp,${omp_min},${omp_max},${omp_step} \
-m=${CFG_OUTPUT} python ${TF_CNN_DIR}/run_single_node_benchmark.py \
--cpu ${cpu} --mkl=True \
--forward_only=${CFG_INFERENCE} --single_socket=${CFG_INFERENCE} \
--num_warmup_batches=0 --distortions=False --optimizer=sgd \
--batch_size=${batch_size} \
--data_format=NCHW --model=${model} \
--num_batches=100 \
--num_omp_threads=\%omp \
--num_inter_threads=\%interop \
--num_intra_threads=\%intraop \
