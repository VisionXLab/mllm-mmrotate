#!/bin/bash

TOTAL_BATCH_SIZE=${TOTAL_BATCH_SIZE:-32}
NGPUS=8
PER_RANK_BATCH_SIZE=$((TOTAL_BATCH_SIZE / NGPUS))

RUN_NAME="florence-2-b_vis1024-lang2048_dior-v2_b2x8xga2-100e-slurm-zero2"

set -x
PYTHONPATH="."$PYTHONPATH ACCELERATE_CPU_AFFINITY=1 torchrun --nproc-per-node=8 \
    -m lmmrotate.train \
    --deepspeed ./lmmrotate/deepspeed_config/zero2.json \
    --model_name_or_path 'microsoft/Florence-2-base' \
    --image_square_length 1024 \
    --language_model_max_length 2048 \
    --data_path ./playground/data/florence-dota/florence_dior_r_trainval_v2.json \
    --image_folder ./playground/data/DIOR/JPEGImages-trainval \
    --bf16 True \
    --attn_implementation "flash_attention_2" \
    --output_dir ./checkpoints/${RUN_NAME} \
    --num_train_epochs 100 \
    --per_device_train_batch_size ${PER_RANK_BATCH_SIZE} \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --eval_strategy "no" \
    --eval_steps 200 \
    --save_strategy "steps" \
    --save_steps 50000 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to "tensorboard" \
    --run_name ${RUN_NAME}
