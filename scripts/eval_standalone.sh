#!/bin/bash
SPLIT=${SPLIT:-"trainval test"}
CKPT=$1
OTHER_ARGS=${@:2}

python -m lmmrotate.eval --model_ckpt_path ${CKPT} --split ${SPLIT} ${OTHER_ARGS}
