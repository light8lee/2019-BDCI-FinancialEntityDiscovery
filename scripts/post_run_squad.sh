#!/bin/bash
name=bert_wwm_squadc
output_dir=outputs/${name}_post
model_dir=outputs/${name}
mkdir -p ${output_dir}
CUDA_VISIBLE_DEVICES=1,3 python post_run_squad.py \
    --train_file=inputs/rc_positive500/dev.json \
    --model_type=bert \
    --model_name_or_path=${model_dir} \
    --output_dir=${output_dir} \
    --max_seq_length=512 \
    --do_train \
    --per_gpu_train_batch_size=20 \
    --num_train_epochs=2 \
    --learning_rate=2e-5 \
    --max_answer_length=15 \
    --n_best_size=1 \
    --overwrite_output_dir \
    --save_steps=1000 &> ${output_dir}/model.log &
