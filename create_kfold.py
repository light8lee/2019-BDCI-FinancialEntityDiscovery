import os
import pandas as pd
import pickle
import numpy as np
import argparse
import create_data

FOLD = 5

def split_kfold(input_dir, output_dir):
    samples = pd.read_csv('./data/Train_Data.csv', sep=',', dtype=str, encoding='utf-8')
    samples.fillna('', inplace=True)
    samples['cleaned_text'] = samples['text'].apply(create_data.clean)
    samples['cleaned_title'] = samples['title'].apply(create_data.clean)

    fold_size = samples.shape[0] // 10
    dev_kfolds = []
    train_kfolds = []

    for k in range(FOLD):
        start = k * fold_size
        if k == FOLD-1:
            end = samples.shape[0]
        else:
            end = (k + 1) * fold_size
        dev_kfolds.append(samples[start:end])
        train_kfolds.append(pd.concat([samples[:start], samples[end:]], ignore_index=True))

    for k in range(FOLD):
        kfold_dir = os.path.join(output_dir, 'fold{}'.format(k))
        if not os.path.isdir(kfold_dir):
            os.mkdir(kfold_dir)
        create_data.create_data(dev_kfolds[k], f'{kfold_dir}/dev.txt', True)
        create_data.create_data(train_kfolds[k], f'{kfold_dir}/train.txt', False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir')
    args = parser.parse_args()
    split_kfold('data', args.output_dir)