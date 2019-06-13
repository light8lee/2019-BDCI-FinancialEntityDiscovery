import models
from proj_utils.files import save_ckpt, load_ckpt, load_config_from_json
from proj_utils.configuration import Config
from proj_utils.logs import log_info
from dataset import GraphDataset, collect_multigraph
import sys
from tqdm import tqdm
import os
import pickle
import argparse
import numpy as np
import torch as t
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from collections import Counter

Precision = lambda tp, fp: tp / (tp + fp)
Recall = lambda tp, fn: tp / (tp + fn)
F1 = lambda p, r: ((2 * p * r) / (p + r)) if (p != 0) and (r != 0) else 0


def infer(data, model, seq_len, cuda):
    features, targets = data
    batch_ids, batch_masks, batch_adjs = features
    labels = targets.numpy()

    if cuda:
        batch_ids = batch_ids.cuda()
        batch_masks = batch_masks.cuda()
        batch_adjs = batch_adjs.cuda()
        targets = targets.cuda()
    log_pred = model(batch_ids, batch_masks, batch_adjs)
    return np.exp(log_pred.cpu().numpy())

def predict(args):
    Log = log_info(os.path.join(args.save_dir, 'kfold.info'))
    Log(args)

    model_config, optimizer_config = Config.from_json(args.config)
    model_name = model_config.name
    model_class = getattr(models, model_name)

    if model_config.init_weight_path is None:
        model_config.init_weight = None
    else:
        model_config.init_weight = t.from_numpy(pickle.load(open(model_config.init_weight_path, 'rb'))).float()

    if model_config.activation is None:
        pass
    elif model_config.activation == 'identical':
        model_config.activation = lambda v: v
    elif model_config.activation == 'gelu':
        model_config.activation = models.layers.activation.gelu
    else:
        model_config.activation = getattr(t, model_config.activation, None) or getattr(F, model_config.activation, None)

    collate_fn = lambda batch: collect_multigraph(model_config.need_norm, batch)

    phase = 'test'
    fea_filename = os.path.join(args.data, '{}.fea'.format(phase))
    tgt_filename = os.path.join(args.data, '{}.tgt'.format(phase))
    pos_filename = os.path.join(args.data, '{}.pos'.format(phase))
    fea_file = open(fea_filename, 'rb')
    with open(tgt_filename, 'r') as f:
        targets = [int(v.strip()) for v in f]
    with open(pos_filename, 'r') as f:
        positions = [int(v.strip()) for v in f]
    dataset = GraphDataset(fea_file, targets, positions)
    dataloader = t.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                            shuffle=False, collate_fn=collate_fn, num_workers=1)

    epochs = args.best_epochs
    epochs = epochs.split(',')
    assert len(epochs) == 10
    total_proba = None
    for fold, epoch in enumerate(epochs):
        model = model_class(**model_config.values)
        ckpt_file = os.path.join(args.save_dir, 'model{}.epoch{}.pt.tar'.format(fold, epoch))
        if os.path.isfile(ckpt_file):
            load_ckpt(ckpt_file, model)
        else:
            raise Exception("No such path {}".format(ckpt_file))
        if args.cuda:
            model = model.cuda()

        model.eval()
        running_loss = 0.
        running_results = Counter()

        curr_proba = []
        pbar = tqdm(dataloader)
        pbar.set_description("[Fold: {}]".format(fold))
        for data in pbar:
            with t.no_grad():
                proba = infer(data, model, model_config.seq_len, args.cuda)
                curr_proba.append(proba)
        curr_proba = np.concatenate(curr_proba, axis=0)
        if total_proba is None:
            total_proba = curr_proba
        else:
            assert total_proba.shape == curr_proba.shape
            total_proba += curr_proba
    
    predictions = total_proba.argmax(1)
    tn, fp, fn, tp = confusion_matrix(dataset.targets, predictions, labels=[0, 1]).ravel()
    precision = Precision(tp, fp)
    recall = Recall(tp, fn)
    acc = (tp + tn) / (tn + fp + fn + tp)
    f1 = F1(precision, recall)
    Log('Acc: {}, P: {}, R: {}, F1: {}'.format(
        acc, precision, recall, f1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('best_epochs', type=str)
    parser.add_argument('--cuda', dest="cuda", action="store_true")
    parser.set_defaults(cuda=False)
    parser.add_argument('--data', type=str, default="./inputs/train", help="input/target data name")
    parser.add_argument('--save_dir', type=str, default='./outputs/', help="model directory path")
    parser.add_argument('--config', type=str, default='model_config.json', help="config file")
    parser.add_argument('--batch_size', type=int, default=16, help="batch size")
    args = parser.parse_args()
    predict(args)
