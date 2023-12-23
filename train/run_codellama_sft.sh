#!/bin/bash

deepspeed_args="--master_port=11000"      # Default argument

# 检查是否提供了正确数量的参数
if [ $# -ne 3 ]; then
    echo "Usage: $0 <model_name_or_path> <data_path> <exp_id>"
    exit 1
fi

project_dir=$(cd "$(dirname $0)"/..; pwd)

model_name_or_path=$1
data_path=$2
exp_id=$3

output_dir=./output_models/${exp_id}

dataset_path=${data_path}

mkdir -p ${output_dir} ${log_dir} d

deepspeed ${deepspeed_args} \
  ./LMFlow/examples/finetune.py \
    --model_name_or_path ${model_name_or_path} \
    --dataset_path ${dataset_path} \
    --output_dir ${output_dir} --overwrite_output_dir \
    --num_train_epochs 2 \
    --learning_rate 2e-5 \
    --disable_group_texts True \
    --overwrite_cache False \
    --block_size 4096 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing 1 \
    --weight_decay 0.1 \
    --deepspeed ./LMFlow/configs/ds_config_zero3.json \
    --bf16 \
    --run_name ${exp_id} \
    --validation_split_percentage 0 \
    --logging_steps 1 \
    --do_train \
    --ddp_timeout 720000 \
    --save_steps 500 \
    --dataloader_num_workers 8 \
    --lr_scheduler_type "cosine" \
    | tee ${log_dir}/train.log \
    2> ${log_dir}/train.err
