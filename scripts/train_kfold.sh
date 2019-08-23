#!/bin/bash
name=kfold_lstm_gat_sdg1t
mkdir -p outputs/${name}
git log | head -6 > outputs/${name}/${name}.info
cp scripts/train_kfold.sh outputs/${name}
cp model_config.json outputs/${name}
data=kfold_inputs_all/stsbenchmark_inputs_dg
for i in {0..9}
do
    CUDA_VISIBLE_DEVICES=2 python train.py \
        --cuda \
        --data=${data} \
        --save_dir=outputs/${name} \
        --batch_size=32 \
        --epoch=20 \
        --config=model_config.json \
        --fold=${i} \
        2> outputs/${name}/${name}.${i}.log >> outputs/${name}/${name}.info &
done