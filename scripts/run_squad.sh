#!/bin/bash
name=bert_squad1
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_squad.py \
    --train_file=inputs/train.json \
    --predict_file=inputs/dev.json \
    --model_type=bert \
    --model_name_or_path=bert_base_chinese \
    --output_dir=outputs/${name} \
    --max_seq_length=128 \
    --do_train \
    --do_eval \
    --per_gpu_train_batch_size=32 \
    --per_gpu_eval_batch_size=32 \
    --num_train_epochs=3 \
    --max_answer_length=20 \
    --n_best_size=1 \
    --overwrite_output_dir \
    --save_steps=1000