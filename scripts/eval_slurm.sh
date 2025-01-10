#!/bin/bash
SPLIT=${SPLIT:-"trainval test"}
CKPT=$1
OTHER_ARGS=${@:2}

NNODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
THEID=$(echo -e $HOSTNAMES | python3 -c "import sys;[sys.stdout.write(str(i)) for i,line in enumerate(next(sys.stdin).split(' ')) if line.strip() == '$(hostname)'.strip()]")
echo MASTER_ADDR=$MASTER_ADDR
echo HOSTNAMES=$HOSTNAMES
echo SLURM_PROCID=$THEID

PYTHONPATH="$PYTHONPATH:$(pwd)" SPLIT=${SPLIT} torchrun --nnodes=$NNODES --nproc-per-node=8 \
    --master_port 12955 --master_addr ${MASTER_ADDR} --node_rank ${THEID} \
    -m lmmrotate.eval --model_ckpt_path ${CKPT} --split ${SPLIT} ${OTHER_ARGS}
