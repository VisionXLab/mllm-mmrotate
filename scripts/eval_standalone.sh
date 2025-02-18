#!/bin/bash
NGPUS=${NGPUS:-1}
SPLIT=${SPLIT:-"trainval test"}
CKPT=$1
OTHER_ARGS=${@:2}

torchrun --nproc_per_node=${NGPUS} -m lmmrotate.eval --model_ckpt_path ${CKPT} --split ${SPLIT} ${OTHER_ARGS}
