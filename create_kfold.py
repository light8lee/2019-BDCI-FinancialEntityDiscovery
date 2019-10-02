import os
import pandas as pd
import pickle
import numpy as np
import argparse
import create_data

FOLD = 5

def split_kfold(input_dir, output_dir):
    train_data = pd.read_csv('./data/Train_Data.csv', sep=',', dtype=str, encoding='utf-8')
    test_data = pd.read_csv('./data/Test_Data.csv', sep=',', dtype=str, encoding='utf-8')

    train_data.fillna('', inplace=True)
    test_data.fillna('', inplace=True)

    train_data['cleaned_text'] = train_data['text'].apply(create_data.clean)
    train_data['cleaned_title'] = train_data['title'].apply(create_data.clean)
    test_data['cleaned_text'] = test_data['text'].apply(create_data.clean)
    test_data['cleaned_title'] = test_data['title'].apply(create_data.clean)

    create_data.remove_chars(train_data, test_data)

    fold_size = train_data.shape[0] // FOLD
    dev_kfolds = []
    train_kfolds = []

    for k in range(FOLD):
        start = k * fold_size
        if k == FOLD-1:
            end = train_data.shape[0]
        else:
            end = (k + 1) * fold_size
        dev_kfolds.append(train_data[start:end])
        train_kfolds.append(pd.concat([train_data[:start], train_data[end:]], ignore_index=True))

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