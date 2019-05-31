#!/bin/bash
name=test
mkdir -p outputs/${name}
git log | head -6 > outputs/${name}/${name}.info
cp scripts/pretrain.sh outputs/${name}
cp model_config.json outputs/${name}
# CUDA_VISIBLE_DEVICES=0,2 python -m torch.distributed.launch --nproc_per_node=2 train.py \
CUDA_VISIBLE_DEVICES=1 python pretrain.py \
    --cuda \
    --name=${name} \
    --data=inputs/ \
    --save_dir=outputs/${name} \
    --batch_size=64 \
    --epoch=100 \
    --config=model_config.json \
    --multi_gpu \
    2> outputs/${name}/${name}.log >> outputs/${name}/${name}.info
#    --conti= \
