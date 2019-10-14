#!/bin/bash
name=roberta_ext_squadc
CUDA_VISIBLE_DEVICES=3 python predict_squad.py \
    --predict_file=inputs/rc_fullv4/test.json \
    --model_type=bert \
    --save_dir=outputs/${name} \
    --pred_batch_size=16 \
    --doc_stride=400 \
    --n_best_size=20 \
    --max_seq_length=512 \
    --max_answer_length=15
