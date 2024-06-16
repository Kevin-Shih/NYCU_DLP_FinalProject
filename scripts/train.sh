#!/usr/bin/env bash
# run this script in the root path of TransMVSNet
MVS_TRAINING="./data/DTU/dtu_training" # path to dataset dtu_training
LOG_DIR="./outputs/dtu_training" # path to checkpoints
if [ ! -d $LOG_DIR ]; then
	mkdir -p $LOG_DIR
fi

NUM_THREADS=1
NGPUS=1
BATCH_SIZE=8
# OMP_NUM_THREADS=$NUM_THREADS python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py \
OMP_NUM_THREADS=$NUM_THREADS python -m torch.distributed.run --nproc_per_node=$NGPUS train.py \
	--logdir=$LOG_DIR \
	--dataset=dtu_yao \
	--batch_size=$BATCH_SIZE \
	--epochs=16 \
	--trainpath=$MVS_TRAINING \
	--trainlist=lists/dtu/train.txt \
	--testlist=lists/dtu/val.txt \
	--numdepth=192 \
	--ndepths="48,32,8" \
	--nviews=5 \
	--wd=0.0001 \
	--depth_inter_r="4.0,1.0,0.5" \
	--lrepochs="6,8,12:2" \
	--dlossw="1.0,1.0,1.0"\
    --summary_freq 20\
    --use_box | tee -a $LOG_DIR/log.txt\
