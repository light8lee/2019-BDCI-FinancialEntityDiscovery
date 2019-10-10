#!/bin/bash
name=bert_wwm_v21k1
output_dir=outputs/${name}
batch_size=16
mkdir -p ${output_dir}
git log | head -6 > ${output_dir}/${name}.info
cp scripts/train_5fold.sh ${output_dir}
cp model_config.json ${output_dir}
for i in {0..1}  # change it
do
    fold_dir=${output_dir}/fold${i}
    mkdir -p ${fold_dir}
    CUDA_VISIBLE_DEVICES=$[$i+1] python train.py \
        --cuda \
        --data=inputs/ner_5foldv4/fold${i} \
        --save_dir=${fold_dir} \
        --batch_size=${batch_size} \
        --epoch=6 \
        --do_eval \
        --config=model_config.json \
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
    CUDA_VISIBLE_DEVICES=$[$i-1] python train.py \
        --cuda \
        --data=inputs/ner_5foldv4/fold${i} \
        --save_dir=${fold_dir} \
        --batch_size=${batch_size} \
        --epoch=6 \
        --do_eval \
        --config=model_config.json \
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
    CUDA_VISIBLE_DEVICES=1 python train.py \
        --cuda \
        --data=inputs/ner_5foldv4/fold${i} \
        --save_dir=${fold_dir} \
        --batch_size=${batch_size} \
        --epoch=6 \
        --do_eval \
        --config=model_config.json \
        2> ${fold_dir}/model.log >> ${fold_dir}/model.info &
done