#!/bin/bash
name=bert_v3
post_name=${name}_post
mkdir -p outputs/${post_name}
git log | head -6 > outputs/${post_name}/${post_name}.info
cp scripts/post_train.sh outputs/${post_name}
cp outputs/${name}/model_config.json outputs/${post_name}/
CUDA_VISIBLE_DEVICES=1 python post_train.py \
    --cuda \
    --data=inputs \
    --save_dir=outputs/${post_name} \
    --load_dir=outputs/${name} \
    --batch_size=32 \
    --epoch=2 \
    --config=outputs/${post_name}/model_config.json \
    2> outputs/${post_name}/${post_name}.log >> outputs/${post_name}/${post_name}.info &
