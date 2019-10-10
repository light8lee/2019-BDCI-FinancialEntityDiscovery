#!/bin/bash
data_dir=inputs/ner_full510v3
mkdir -p ${data_dir}
python create_data.py ${data_dir}

python preprocess_task.py ${data_dir} ${data_dir} \
    bert_wwm_ext/vocab.txt \
    --max_seq_length=510 --need_dev