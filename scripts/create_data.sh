#!/bin/bash
mkdir -p inputs/ner_full510
python create_data.py inputs/ner_full510

python preprocess_task.py inputs/ner_full510/ inputs/ner_full510/ \
    bert_wwm_ext/vocab.txt \
    --max_seq_length=510 --need_dev