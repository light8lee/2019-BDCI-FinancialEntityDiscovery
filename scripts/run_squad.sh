#!/bin/bash
name=roberta_ext_squadd
output_dir=outputs/${name}
mkdir -p ${output_dir}
cp scripts/run_squad.sh ${output_dir}/
CUDA_VISIBLE_DEVICES=3 python run_squad.py \
    --train_file=inputs/rc_fullv4/train.json \
    --predict_file=inputs/rc_fullv4/dev.json \
    --model_type=bert \
    --model_name_or_path=roberta_zh_ext \
    --output_dir=${output_dir} \
    --max_seq_length=512 \
    --doc_stride=400 \
    --do_train \
    --do_eval \
    --per_gpu_train_batch_size=16 \
    --per_gpu_eval_batch_size=16 \
    --num_train_epochs=4 \
    --learning_rate=2e-5 \
    --max_answer_length=15 \
    --n_best_size=2 \
    --overwrite_output_dir \
    --save_steps=1000 &> ${output_dir}/model.log &
