#!/bin/bash
name=bert_squad1
CUDA_VISIBLE_DEVICES=3 python predict_squad.py \
    --predict_file=inputs/test.json \
    --model_type=bert \
    --save_dir=outputs/${name} \
    --pred_batch_size=64 \
    --max_seq_length=128 \
    --max_answer_length=20