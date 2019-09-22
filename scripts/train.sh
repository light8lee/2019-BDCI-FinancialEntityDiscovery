#!/bin/bash
name=bert_wwm_v13
mkdir -p outputs/${name}
git log | head -6 > outputs/${name}/${name}.info
cp scripts/train.sh outputs/${name}
cp model_config.json outputs/${name}
# CUDA_VISIBLE_DEVICES=1,2,3 python -m torch.distributed.launch --nproc_per_node=3 train.py \
CUDA_VISIBLE_DEVICES=0 python train.py \
    --cuda \
    --data=inputs \
    --save_dir=outputs/${name} \
    --batch_size=64 \
    --epoch=4 \
    --do_eval \
    --config=model_config.json \
    2> outputs/${name}/${name}.log >> outputs/${name}/${name}.info &
    # --multi_gpu \
#    --conti= \
