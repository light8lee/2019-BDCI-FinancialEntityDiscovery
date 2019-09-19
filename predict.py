import models
from sklearn.externals import joblib
from proj_utils.files import save_ckpt, load_ckpt, load_config_from_json
from proj_utils.configuration import Config
from proj_utils.logs import log_info
from dataset import GraphDataset, collect_single
import sys
from tqdm import tqdm
import os
import pandas as pd
import pickle
import argparse
import numpy as np
import torch as t
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from collections import Counter, defaultdict
from scipy.stats import pearsonr
from task_metric import get_BIO_entities
from tokenization import convert_ids_to_tokens, load_vocab


def infer(data, model, inv_vocabs, cuda):
    idxs, batch_ids, batch_masks, batch_tags, batch_inputs = data
    print(idxs)

    if cuda:
        if isinstance(batch_ids, t.Tensor):
            batch_ids_t = batch_ids.cuda()
            batch_masks_t = batch_masks.cuda()
        else:
            batch_ids_t = [v.cuda() for v in batch_ids]
            batch_masks_t = [v.cuda() for v in batch_masks]
    batch_lens = batch_masks_t.sum(-1).tolist()
    pred_tags = model.predict(batch_ids_t, batch_masks_t)
    results = defaultdict(set)
    for idx, entities, inputs in zip(idxs, get_BIO_entities(pred_tags, batch_lens), batch_inputs):
        results[idx].add('')
        for start, end in entities:
            result = ''.join(inputs[start:end])
            result = result.replace('※', ' ')
            results[idx].add(result)
    return results


def predict(args):
    vocabs = load_vocab(args.vocab)
    inv_vocabs = {v: k for k, v in vocabs.items()}
    model_config, optimizer_config, _ = Config.from_json(args.config)
    model_name = model_config.name
    model_class = getattr(models, model_name)

    if model_config.init_weight_path is None:
        model_config.init_weight = None
    else:
        model_config.init_weight = t.from_numpy(pickle.load(open(model_config.init_weight_path, 'rb'))).float()

    phase = 'test'
    fea_filename = os.path.join(args.data, '{}.fea'.format(phase))
    pos_filename = os.path.join(args.data, '{}.pos'.format(phase))
    fea_file = open(fea_filename, 'rb')
    with open(pos_filename, 'r') as f:
        positions = [int(v.strip()) for v in f]
    dataset = GraphDataset(fea_file, positions)
    dataloader = t.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                            shuffle=False, collate_fn=collect_single, num_workers=1)

    model = model_class(**model_config.values)
    ckpt_file = os.path.join(args.save_dir, 'model.{}.pt.tar'.format(args.model))
    if os.path.isfile(ckpt_file):
        load_ckpt(ckpt_file, model)
    else:
        raise Exception("No such path {}".format(ckpt_file))
    if args.cuda:
        model = model.cuda()

    model.eval()
    curr_preds = defaultdict(set)
    pbar = tqdm(dataloader)
    for data in pbar:
        with t.no_grad():
            results = infer(data, model, inv_vocabs, args.cuda)
            for key in results:
                curr_preds[key].update(results[key])
    idxs = []
    entities = []
    for key in curr_preds:
        print(key)
        idxs.append(key)
        curr_preds[key].remove('')
        entities.append(';'.join(curr_preds[key]))
    preds = pd.DataFrame({'id': idxs, 'unknownEntities': entities})
    preds.to_csv(os.path.join(args.save_dir, 'submit.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab', type=str, default='bert_model/vocab.txt')
    parser.add_argument('--model', type=str, default='best')
    parser.add_argument('--cuda', dest="cuda", action="store_true")
    parser.set_defaults(cuda=False)
    parser.add_argument('--data', type=str, default="./inputs/train", help="input/target data name")
    parser.add_argument('--save_dir', type=str, default='./outputs/', help="model directory path")
    parser.add_argument('--config', type=str, default='model_config.json', help="config file")
    parser.add_argument('--batch_size', type=int, default=16, help="batch size")
    parser.add_argument('--seed', type=int, default=2019)
    args = parser.parse_args()
    t.manual_seed(args.seed)
    t.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    predict(args)
