#!/bin/bash
name=bert_wwm_squadh
CUDA_VISIBLE_DEVICES=3 python predict_squad.py \
    --predict_file=inputs/rc_fullv3/test.json \
    --model_type=bert \
    --save_dir=outputs/${name} \
    --pred_batch_size=32 \
    --max_seq_length=512 \
    --max_answer_length=15
