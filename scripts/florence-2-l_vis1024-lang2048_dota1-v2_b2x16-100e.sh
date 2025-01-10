#!/bin/bash
TOTAL_BATCH_SIZE=${TOTAL_BATCH_SIZE:-32}
NGPUS=16
PER_RANK_BATCH_SIZE=$((TOTAL_BATCH_SIZE / NGPUS))
RUN_NAME="florence-2-l_vis1024-lang2048_dota1-v2_b2x16-100e-zero2"

set -x
PYTHONPATH="."$PYTHONPATH ACCELERATE_CPU_AFFINITY=1 torchrun --nnodes=2 --nproc-per-node=8 \
    --master_port 12968 --master_addr ${MASTER_ADDR} --node_rank ${THEID} \
    -m lmmrotate.train \
    --deepspeed ./lmmrotate/deepspeed_config/zero2.json \
    --model_name_or_path 'microsoft/Florence-2-large' \
    --image_square_length 1024 \
    --language_model_max_length 2048 \
    --data_path ./playground/data/florence-dota/florence_split_ss_dota_trainval_v2.json \
    --image_folder ./playground/data/split_ss_dota/trainval/images \
    --bf16 True \
    --attn_implementation "flash_attention_2" \
    --output_dir ./checkpoints/${RUN_NAME} \
    --num_train_epochs 100 \
    --per_device_train_batch_size ${PER_RANK_BATCH_SIZE} \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
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
