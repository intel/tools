#!/bin/bash
set -e

if [ $# -ne 2 ];
then
  echo "run_tuna_mkl.sh <model> <cpu>"
  echo "<cpu> options: skl"
  echo "<model> options come from running this: python run_single_node_benchmark.py -h"
  exit 0
fi

model=$1
cpu=$2

#############################
# Configs

HOME_DIR=${HOME:-/home/user}
LOG_DIR=${LOG_DIR:-/tmp}
TF_CNN_DIR=${TF_CNN_DIR:-${HOME_DIR}/private-tensorflow-benchmarks/scripts/tf_cnn_benchmarks}
# enables checking for python script error
CHECK_FOR_ERROR=${CHECK_FOR_ERROR:-False}

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

batch_size_min=8
batch_size_max=2048
batch_size_step=8

if [ "$CFG_INFERENCE" = "True" ];
then
  LOG=${LOG_DIR}/${cpu}/${CFG_OUTPUT}/${CFG_SEARCH}/mkl/${model}.inference.tuna.${CFG_SEARCH}.log
else
  LOG=${LOG_DIR}/${cpu}/${CFG_OUTPUT}/${CFG_SEARCH}/mkl/${model}.train.tuna.${CFG_SEARCH}.log
fi

# tuna doesn't throw an error if the underlying script were to break, so we verify it can run once
# TODO: is there a way tuna can do this for us??
if [ "$CHECK_FOR_ERROR" = "True" ];
then
  echo "Trying underlying script, checking for error..."
  set -x
  python ${TF_CNN_DIR}/run_single_node_benchmark.py \
  --cpu ${cpu} --mkl=True \
  --forward_only=${CFG_INFERENCE} --single_socket=${CFG_INFERENCE} \
  --num_warmup_batches=0 --distortions=False --optimizer=sgd \
  --data_format=NCHW --model=${model} \
  --num_batches=100 \
  --num_omp_threads=${omp_min} \
  --num_inter_threads=${inter_op_min} \
  --num_intra_threads=${intra_op_min} \
  --batch_size=${batch_size_min}
  AA
fi

INIT_RADIUS=${CFG_INIT_RADIUS} \
STRATEGY=${CFG_SEARCH}.so LAYERS=log.so \
LOG_FILE=${LOG} \
${HARMONY_HOME}/bin/tuna \
-q -v -n=200 \
-i=interop,${inter_op_min},${inter_op_max},${inter_op_step} \
-i=intraop,${intra_op_min},${intra_op_max},${intra_op_step} \
-i=omp,${omp_min},${omp_max},${omp_step} \
-i=batch_size,${batch_size_min},${batch_size_max},${batch_size_step} \
-m=${CFG_OUTPUT} python ${TF_CNN_DIR}/run_single_node_benchmark.py \
--cpu ${cpu} --mkl=True \
--forward_only=${CFG_INFERENCE} --single_socket=${CFG_INFERENCE} \
--num_warmup_batches=0 --distortions=False --optimizer=sgd \
--data_format=NCHW --model=${model} \
--num_batches=100 \
--num_omp_threads=%omp \
--num_inter_threads=%interop \
--num_intra_threads=%intraop \
--batch_size=$batch_size
