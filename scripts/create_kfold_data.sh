#!/bin/bash
kfold_dir=inputs/ner_5foldv4
mkdir -p ${kfold_dir}
python create_kfold.py ${kfold_dir}
for i in {0..4}
do
    python preprocess_task.py ${kfold_dir}/fold${i} ${kfold_dir}/fold${i} \
        bert_wwm_ext/vocab.txt \
        --max_seq_length=510 --need_dev --no_test
done