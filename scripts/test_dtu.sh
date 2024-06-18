#!/usr/bin/env bash
TESTPATH="./data/DTU/dtu_test" 						            # path to dataset dtu_test
TESTLIST="./lists/dtu/test.txt"
CKPT_FILE="./outputs/dtu_training_BoxAtten/model_000015.ckpt"   # path to checkpoint file, you need to use the model_dtu.ckpt for testing
FUSIBLE_PATH="./fusibile/fusibile"                                # path to fusible of gipuma
OUTDIR="./outputs/dtu_testing_BoxAtten2" 						    # path to output ./outputs/dtu_testing_BoxAtten
if [ ! -d $OUTDIR ]; then
	mkdir -p $OUTDIR
fi

python test.py \
    --dataset=general_eval \
    --batch_size=1 \
    --testpath=$TESTPATH  \
    --testlist=$TESTLIST \
    --loadckpt=$CKPT_FILE \
    --outdir=$OUTDIR \
    --numdepth=192 \
    --ndepths="48,32,8" \
    --num_view=5 \
    --depth_inter_r="4.0,1.0,0.5" \
    --interval_scale=1.06 \
    --use_box \
    --filter_method="gipuma" \
    --fusibile_exe_path=$FUSIBLE_PATH
    # --use_mscale
    # --filter_method="normal"