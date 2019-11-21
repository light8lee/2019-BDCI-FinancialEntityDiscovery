#!/bin/bash
max_seq_len=128
data_dir=round2_inputs/ner_mrc_L${max_seq_len}_F5_NoLM
mkdir -p ${data_dir}
python create_mrc_data.py \
    ${data_dir} \
    --max_seq_len=${max_seq_len}

python preprocess_mrc_task.py ${data_dir} ${data_dir} \
    roberta_zh_ext/vocab.txt \
    --max_seq_length=${max_seq_len} \
    --need_dev \
    --duplicate=1
    # --duplicate=2 \
    # --augment
