#!/bin/bash
name=rnn_diffpool_v2a
mkdir -p outputs/${name}
git log | head -6 > outputs/${name}/${name}.info
cp scripts/train.sh outputs/${name}
cp model_config.json outputs/${name}
for i in {0..9}
do
    CUDA_VISIBLE_DEVICES=1 nohup python train.py \
        --name=${name} \
        --cuda \
        --data=inputs/ \
        --save_dir=outputs/${name} \
        --batch_size=64 \
        --epoch=80 \
        --config=model_config.json \
        --fold=${i} \
        2> outputs/${name}/${name}.${i}.log >> outputs/${name}/${name}.info &

done