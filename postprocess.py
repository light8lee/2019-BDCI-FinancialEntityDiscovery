import pandas as pd
import argparse
import os

def convert_to_submit(name):
    input_filename = os.path.join('outputs', name, 'submit.csv')
    preds = pd.read_csv(input_filename, sep=',', index_col='id')
    sample = pd.read_csv('data/Test_Data.csv', sep=',', index_col='id')
    assert sample.shape[0] == preds.shape[0]
    assert sample.shape[0] == len(sample.index & preds.index)
    print('not in sample', set(preds.index)-set(sample.index))
    print('not in preds', set(sample.index)-set(preds.index))
    outputs = preds.reindex(sample.index)
    output_filename = os.path.join('submits', '{}.csv'.format(name))
    outputs.to_csv(output_filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str)
    args = parser.parse_args()
    convert_to_submit(args.name)