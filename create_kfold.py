import os
import pickle
import numpy as np

def split_kfold(input_dir):
    with open(os.path.join(input_dir, 'train.txt'), 'r', encoding='utf-8') as f:
        samples = f.readlines()
    with open(os.path.join(input_dir, 'dev.txt'), 'r', encoding='utf-8') as f:
        samples.extend(f.readlines())
    fold_size = len(samples) // 10
    dev_kfolds = []
    train_kfolds = []

    for k in range(10):
        start = k * fold_size
        if k == 9:
            end = len(samples)
        else:
            end = (k + 1) * fold_size
        dev_kfolds.append(samples[start:end])
        train_kfolds.append(samples[:start] + samples[end:])

    for k in range(10):
        kfold_dir = os.path.join(input_dir, 'fold{}'.format(k))
        if not os.path.isdir(kfold_dir):
            os.mkdir(kfold_dir)
        with open(os.path.join(kfold_dir, 'dev.txt'), 'w', encoding='utf-8') as f:
            f.writelines(dev_kfolds[k])
        with open(os.path.join(kfold_dir, 'train.txt'), 'w', encoding='utf-8') as f:
            f.writelines(train_kfolds[k])

if __name__ == '__main__':
    split_kfold('stsbenchmark')