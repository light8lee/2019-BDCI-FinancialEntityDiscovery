#!/bin/bash
name=bert_wwm_squadb
CUDA_VISIBLE_DEVICES=2 python predict_squad.py \
    --predict_file=inputs/rc_positive/test.json \
    --model_type=bert \
    --save_dir=outputs/${name} \
    --pred_batch_size=64 \
    --max_seq_length=128 \
    --max_answer_length=15
