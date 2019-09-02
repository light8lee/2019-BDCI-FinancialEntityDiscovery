import models
from sklearn.externals import joblib
from proj_utils.files import save_ckpt, load_ckpt, load_config_from_json
from proj_utils.configuration import Config
from proj_utils.logs import log_info
from dataset import GraphDataset, collect_single
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
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix
from collections import Counter
import pdb
from scipy.stats import pearsonr
import task_metric as tm


def infer(data, model, cuda):
    idx, batch_ids, batch_masks, batch_tags, _ = data

    if cuda:
        if isinstance(batch_ids, t.Tensor):
            batch_ids = batch_ids.cuda()
            batch_masks = batch_masks.cuda()
            batch_tags = batch_tags.cuda()
        else:
            batch_ids = [v.cuda() for v in batch_ids]
            batch_masks = [v.cuda() for v in batch_masks]
            batch_tags = [v.cuda() for v in batch_tags]

    loss = model(batch_ids, batch_masks, batch_tags)
    size = batch_masks.shape[0]
    return size, -loss


def train(args):
    Log = log_info(os.path.join(args.save_dir, 'process.info'))
    Log(args)
    model_config, optimizer_config, scheduler_config = Config.from_json(args.config)
    model_name = model_config.name
    model_class = getattr(models, model_name)

    if model_config.init_weight_path is None:
        model_config.init_weight = None
    else:
        model_config.init_weight = t.from_numpy(pickle.load(open(model_config.init_weight_path, 'rb'))).float()

    model = model_class(**model_config.values)

    phase == 'dev'
    dataloaders = {}
    datasets = {}
    sampler = None
    if model_config.name.find("BERT") != -1:
        collate_fn = collect_single
    else:
        collate_fn = collect_multigraph
        fea_filename = os.path.join(args.data, '{}.fea'.format(phase))
        pos_filename = os.path.join(args.data, '{}.pos'.format(phase))
        fea_file = open(fea_filename, 'rb')
        with open(pos_filename, 'r') as f:
            positions = [int(v.strip()) for v in f]
        dataset = GraphDataset(fea_file, positions)
        dataloader = t.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                            shuffle=True, collate_fn=collate_fn, num_workers=1)
        dataloaders[phase] = dataloader
        datasets[phase] = dataset

    if model_config.name.find("BERT") != -1:
        if model_config.freeze:
            for param in model.bert4pretrain.parameters():
                param.requires_grad = False
        optimizer = getattr(optim, optimizer_config.name)(model.parameters(), **optimizer_config.values)
    else:
        optimizer = getattr(optim, optimizer_config.name)(model.parameters(), **optimizer_config.values)
    scheduler = getattr(optim.lr_scheduler, scheduler_config.name)(optimizer, **scheduler_config.values)

    if not os.path.isdir(args.load_dir):
        os.mkdir(args.load_dir)
    ckpt_file = os.path.join(args.load_dir, 'model.best.pt.tar')
    if os.path.isfile(ckpt_file):
        load_ckpt(ckpt_file, model, optimizer, scheduler)
        conti = args.conti + 1
    else:
        raise Exception("No such path {}".format(ckpt_file))

    if args.cuda:
        model = model.cuda()

    # pdb.set_trace()

    for epoch in range(1, 1+args.epoch):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()

        pbar = tqdm(dataloaders[phase])
        pbar.set_description("[{} Epoch {}]".format(phase, epoch))
        running_loss = 0.
        runnning_size = 0.
        for data in pbar:
            optimizer.zero_grad()

            size, loss = infer(data, model, args.cuda)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_size += size
            pbar.set_postfix(mean_loss=running_loss/running_size)
    save_ckpt(os.path.join(args.save_dir, 'model.best.pt.tar'),
                            epoch, model.state_dict(), optimizer.state_dict(), scheduler.state_dict())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', dest="cuda", action="store_true")
    parser.set_defaults(cuda=False)
    parser.add_argument('--log', dest='log', action='store_true', help='whether use tensorboard')
    parser.add_argument('--data', type=str, default="./inputs/train", help="input/target data name")
    parser.add_argument('--save_dir', type=str, default='./outputs/', help="model directory path")
    parser.add_argument('--load_dir', type=str, default='./outputs/', help="model directory path")
    parser.add_argument('--config', type=str, default='model_config.json', help="config file")
    parser.add_argument('--batch_size', type=int, default=16, help="batch size")
    parser.add_argument('--epoch', type=int, default=20, help="number of epochs")
    parser.add_argument('--seed', type=int, default=2019)
    parser.set_defaults(multi_gpu=False, log=False, do_test=False)
    args = parser.parse_args()
    t.manual_seed(args.seed)
    t.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    train(args)
