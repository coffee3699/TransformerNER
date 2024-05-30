#!/bin/bash

PROJECT_ROOT=$(pwd)
export PYTHONPATH=$PROJECT_ROOT

export CUDA_VISIBLE_DEVICES=0,1

python3 run_ner.py \
  --model_name_or_path /data/zzy/Models/bert-base-chinese \
  --trust_remote_code \
  --use_custom_dataset \
  --dataset_script utils/ner_dataset.py \
  --output_dir output \
  --do_train \
  --do_eval \
  --do_predict \
  --per_device_train_batch_size 64 \
  --per_device_eval_batch_size 64 \
  --eval_strategy "epoch" \
  --save_strategy "epoch" \
  --fp16 \
  --optim "adamw_torch" \
  --logging_steps 500 \
  --num_train_epochs 5 \
  --learning_rate 5e-5 \
  --lr_scheduler_type "linear" \
  --warmup_ratio 0.0 \
  --overwrite_output_dir