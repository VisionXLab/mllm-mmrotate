#!/bin/bash
RUN_NAME="florence-2-l_vis1024-lang2048_dota1-v2_b1x1xga32-50e"

set -x
ACCELERATE_CPU_AFFINITY=1 python -m lmmrotate.train \
    --model_name_or_path 'microsoft/Florence-2-large' \
    --image_square_length 1024 \
    --language_model_max_length 2048 \
    --data_path ./playground/data/florence-dota/florence_split_ss_dota_trainval_v2.json \
    --image_folder ./playground/data/split_ss_dota/trainval/images \
    --fp16 True \
    --attn_implementation "flash_attention_2" \
    --output_dir ./checkpoints/${RUN_NAME} \
    --num_train_epochs 50 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --eval_strategy "no" \
    --eval_steps 200 \
    --save_strategy "steps" \
    --save_steps 50000 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing False \
    --dataloader_num_workers 4 \
    --report_to "tensorboard" \
    --run_name ${RUN_NAME}
