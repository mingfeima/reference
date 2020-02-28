#!/bin/bash

set -e

#export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000";
#export LD_PRELOAD=/home/mingfeim/packages/jemalloc-5.2.0/lib/libjemalloc.so
#export LD_PRELOAD=/home/mingfeim/packages/gperftools-2.7/install/lib/libtcmalloc.so
#export LD_PRELOAD=/home/mingfeim/anaconda3/envs/pytorch-mingfei/lib/libtbbmalloc_proxy.so.2

CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
TOTAL_CORES=`expr $CORES \* $SOCKETS`

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"
KMP_BLOCKTIME=1

export $KMP_SETTING
export KMP_BLOCKTIME=$KMP_BLOCKTIME
export OMP_NUM_THREADS=$TOTAL_CORES
LAST_CORE=`expr $CORES - 1`

echo -e "\n### using $KMP_SETTING"
echo -e "### using KMP_BLOCKTIME=$KMP_BLOCKTIME"
echo -e "### using OMP_NUM_THREADS=$TOTAL_CORES"

#PREFIX="numactl --physcpubind=0-$LAST_CORE --membind=0"
#echo -e "### using $PREFIX\n\n"

DATASET_DIR='../data'

SEED=${1:-"1"}
TARGET=${2:-"24.00"}

# run training
$PREFIX python3 train.py \
  --dataset-dir ${DATASET_DIR} \
  --seed $SEED \
  --target-bleu $TARGET \
  --no-cuda \
  --train-loader-workers 0 \
  --val-loader-workers 0
