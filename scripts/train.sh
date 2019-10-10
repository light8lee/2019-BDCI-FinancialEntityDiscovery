#!/bin/bash
name=bert_wwm_v52
mkdir -p outputs/${name}
git log | head -6 > outputs/${name}/${name}.info
cp scripts/train.sh outputs/${name}
cp model_config.json outputs/${name}
CUDA_VISIBLE_DEVICES=2 python train.py \
    --cuda \
    --data=inputs/ner_full510v3 \
    --save_dir=outputs/${name} \
    --batch_size=16 \
    --epoch=8 \
    --do_eval \
    --config=model_config.json \
    --multi_gpu \
    2> outputs/${name}/${name}.log >> outputs/${name}/${name}.info &
#    --conti= \
