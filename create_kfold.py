import os
import pandas as pd
import pickle
import numpy as np
import argparse
import create_data

FOLD = 5

def split_kfold(input_dir, output_dir, keep_none):
    train_data = pd.read_csv('./data/Train_Data.csv', sep=',', dtype=str, encoding='utf-8')
    test_data = pd.read_csv('./data/Test_Data.csv', sep=',', dtype=str, encoding='utf-8')

    train_data.fillna('', inplace=True)
    test_data.fillna('', inplace=True)

    train_data['cleaned_text'] = train_data['text'].apply(create_data.clean)
    train_data['cleaned_title'] = train_data['title'].apply(create_data.clean)
    test_data['cleaned_text'] = test_data['text'].apply(create_data.clean)
    test_data['cleaned_title'] = test_data['title'].apply(create_data.clean)

    important_chars = create_data.collect_important_chars(train_data['unknownEntities'])

    create_data.remove_chars(train_data, test_data)

    dev_kfolds = []
    train_kfolds = []

    for k in range(FOLD):
        train_data = train_data.sample(frac=1, random_state=2018-k).reset_index(drop=True)
        dev_kfolds.append(train_data.tail(100))
        train_kfolds.append(train_data.head(train_data.shape[0]-100))

    for k in range(FOLD):
        kfold_dir = os.path.join(output_dir, 'fold{}'.format(k))
        if not os.path.isdir(kfold_dir):
            os.mkdir(kfold_dir)
        create_data.create_data(dev_kfolds[k], f'{kfold_dir}/dev.txt', important_chars, True, keep_none)
        create_data.create_data(train_kfolds[k], f'{kfold_dir}/train.txt', important_chars, False, keep_none)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir')
    parser.add_argument('--keep_none', default=False, action='store_true')
    args = parser.parse_args()
    split_kfold('data', args.output_dir, args.keep_none)