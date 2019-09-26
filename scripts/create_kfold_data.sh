#!/bin/bash
python create_kfold.py inputs/ner_5fold
for i in {0..4}
do
    python preprocess_task.py inputs/ner_5fold/fold${i} inputs/ner_5fold/fold${i} \
        bert_wwm_ext/vocab.txt \
        --max_seq_length=500 --need_dev --no_test
done