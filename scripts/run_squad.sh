#!/bin/bash
name=bert_squad3
output_dir=outputs/${name}
mkdir -p ${output_dir}
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_squad.py \
    --train_file=inputs/train.json \
    --predict_file=inputs/dev.json \
    --model_type=bert \
    --model_name_or_path=bert_wwm_ext \
    --output_dir=${output_dir} \
    --max_seq_length=128 \
    --do_train \
    --do_eval \
    --per_gpu_train_batch_size=32 \
    --per_gpu_eval_batch_size=32 \
    --num_train_epochs=3 \
    --learning_rate=4e-5 \
    --max_answer_length=15 \
    --n_best_size=1 \
    --overwrite_output_dir \
    --save_steps=1000 &> ${output_dir}/model.log &