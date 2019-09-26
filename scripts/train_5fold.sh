#!/bin/bash
name=bert_wwm_vh
output_dir=outputs/${name}
mkdir -p ${output_dir}
git log | head -6 > ${output_dir}/${name}.info
cp scripts/train_5fold.sh ${output_dir}
cp model_config.json ${output_dir}
for i in {0..1}  # change it
do
    fold_dir=${output_dir}/fold${i}
    mkdir -p ${fold_dir}
    CUDA_VISIBLE_DEVICES=$[2*$i],$[2*$i+1] python train.py \
        --cuda \
        --data=inputs/ner_5fold/fold${i} \
        --save_dir=${fold_dir} \
        --batch_size=24 \
        --epoch=4 \
        --do_eval \
        --config=model_config.json \
        --multi_gpu \
        2> ${fold_dir}/model.log >> ${fold_dir}/model.info &
done
for job in `jobs -p`; do
    echo Wait on $job
    wait $job
done
for i in {2..3}  # change it
do
    fold_dir=${output_dir}/fold${i}
    mkdir -p ${fold_dir}
    CUDA_VISIBLE_DEVICES=$[2*$i-4],$[2*$i-3] python train.py \
        --cuda \
        --data=inputs/ner_5fold/fold${i} \
        --save_dir=${fold_dir} \
        --batch_size=24 \
        --epoch=4 \
        --do_eval \
        --config=model_config.json \
        --multi_gpu \
        2> ${fold_dir}/model.log >> ${fold_dir}/model.info &
done
for job in `jobs -p`; do
    echo Wait on $job
    wait $job
done
for i in {4..4}  # change it
do
    fold_dir=${output_dir}/fold${i}
    mkdir -p ${fold_dir}
    CUDA_VISIBLE_DEVICES=0,1 python train.py \
        --cuda \
        --data=inputs/ner_5fold/fold${i} \
        --save_dir=${fold_dir} \
        --batch_size=24 \
        --epoch=4 \
        --do_eval \
        --config=model_config.json \
        --multi_gpu \
        2> ${fold_dir}/model.log >> ${fold_dir}/model.info &
done