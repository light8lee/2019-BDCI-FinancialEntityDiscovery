### 
# @Description: 
 # @Author: light8lee
 # @Github: https://github.com/light8lee
 # @Date: 2019-11-07 17:51:37
 # @LastEditors: light8lee
 # @LastEditTime: 2019-11-07 17:52:32
 ###
#!/bin/bash
name=roberta_ext_m11
mkdir -p outputs/${name}
git log | head -6 > outputs/${name}/${name}.info
cp scripts/train.sh outputs/${name}
config=mrc_model_config.json
cp ${config} outputs/${name}
CUDA_VISIBLE_DEVICES=0 python train_mrc.py \
    --cuda \
    --data=round2_inputs/ner_mrc_L256_F5_NoLM2 \
    --save_dir=outputs/${name} \
    --batch_size=8 \
    --scale_rate=10 \
    --epoch=2 \
    --do_eval \
    --config=outputs/${name}/${config} \
    2> outputs/${name}/${name}.log >> outputs/${name}/${name}.info &
    # --multi_gpu \
#    --conti= \
