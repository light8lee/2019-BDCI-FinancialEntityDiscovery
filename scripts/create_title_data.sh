#!/bin/bash
max_seq_len=256
data_dir=round2_inputs/nere_title_T3_L${max_seq_len}_F5_LM
mkdir -p ${data_dir}
python create_title_data.py \
    ${data_dir} \
    --max_seq_len=${max_seq_len}

python preprocess_task.py ${data_dir} ${data_dir} \
    /raid/lt/pretrained_models/ernie_v1/vocab.txt \
    --max_seq_length=${max_seq_len} \
    --need_dev \
    --duplicate=2 \
    --augment
