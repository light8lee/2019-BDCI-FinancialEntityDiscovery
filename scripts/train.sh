### 
# @Description: 
 # @Author: light8lee
 # @Github: https://github.com/light8lee
 # @Date: 2019-11-07 17:51:37
 # @LastEditors: light8lee
 # @LastEditTime: 2019-11-07 17:52:32
 ###
#!/bin/bash
name=roberta_ext_v25
mkdir -p outputs/${name}
# git log | head -6 > outputs/${name}/${name}.info
cp scripts/train.sh outputs/${name}
config=model_config.json
cp ${config} outputs/${name}
CUDA_VISIBLE_DEVICES=0 python train.py \
    --cuda \
    --data=round2_inputs/ner_T3_L510_F8_LM \
    --tag_type=BIO \
    --save_dir=outputs/${name} \
    --batch_size=8 \
    --scale_rate=10 \
    --epoch=4 \
    --save_steps=1000 \
    --do_eval \
    --config=outputs/${name}/${config} \
    2> outputs/${name}/${name}.log >> outputs/${name}/${name}.info &
    # --multi_gpu \
#    --conti= \
