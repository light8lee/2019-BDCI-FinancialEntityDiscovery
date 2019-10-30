#!/bin/bash
max_seq_len=510
data_dir=inputs/ner_T3_L${max_seq_len}_F8_LM
mkdir -p ${data_dir}
python create_data.py \
    ${data_dir} \
    --max_seq_len=${max_seq_len}

python preprocess_task.py ${data_dir} ${data_dir} \
    bert_wwm_ext/vocab.txt \
    --max_seq_length=${max_seq_len} \
    --need_dev \
    --duplicate=2 \
    --augment
