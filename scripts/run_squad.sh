#!/bin/bash
CUDA_VISIBLE_DEVICES=2,3 python run_squad.py \
    --train_file=inputs/train.json \
    --predict_file=inputs/dev.json \
    --model_type=bert \
    --model_name_or_path=bert_base_chinese \
    --output_dir=outputs/squad/ \
    --max_seq_length=128 \
    --do_train \
    --do_eval \
    --save_steps=500