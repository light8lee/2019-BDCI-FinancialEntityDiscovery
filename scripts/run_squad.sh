#!/bin/bash
name=bert_wwm_squadb
output_dir=outputs/${name}
mkdir -p ${output_dir}
CUDA_VISIBLE_DEVICES=1,2 python run_squad.py \
    --train_file=inputs/rc_positive/train.json \
    --predict_file=inputs/rc_positive/dev.json \
    --model_type=bert \
    --model_name_or_path=bert_wwm_ext \
    --output_dir=${output_dir} \
    --max_seq_length=128 \
    --do_train \
    --do_eval \
    --per_gpu_train_batch_size=64 \
    --per_gpu_eval_batch_size=64 \
    --num_train_epochs=4 \
    --learning_rate=4e-5 \
    --max_answer_length=15 \
    --n_best_size=1 \
    --overwrite_output_dir \
    --save_steps=1000 &> ${output_dir}/model.log &
