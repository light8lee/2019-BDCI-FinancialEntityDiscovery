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
    batch_ids, batch_masks, batch_tags = data

    if cuda:
        if isinstance(batch_ids, t.Tensor):
            batch_ids = batch_ids.cuda()
            batch_masks = batch_masks.cuda()
            batch_tags = batch_tags.cuda()
        else:
            batch_ids = [v.cuda() for v in batch_ids]
            batch_masks = [v.cuda() for v in batch_masks]
            batch_tags = [v.cuda() for v in batch_tags]

    emissions, loss = model(batch_ids, batch_masks, batch_tags)
    result = {
        'target_tag_ids': batch_tags,
        'pred_tag_ids': model.decode(emissions, batch_masks),
        'max_lens': batch_masks.sum(-1).tolist(),
        'batch_size': batch_masks.shape[0]
    }
    return result, -loss


def train(args):
    Log = log_info(os.path.join(args.save_dir, 'process{}.info'.format(args.fold)))
    Log(args)
    model_config, optimizer_config, scheduler_config = Config.from_json(args.config)
    model_name = model_config.name
    model_class = getattr(models, model_name)

    if model_config.init_weight_path is None:
        model_config.init_weight = None
    else:
        model_config.init_weight = t.from_numpy(pickle.load(open(model_config.init_weight_path, 'rb'))).float()

    model = model_class(**model_config.values)

    dataloaders = {}
    datasets = {}
    sampler = None
    if model_config.name.find("BERT") != -1:
        collate_fn = collect_single
    else:
        collate_fn = collect_multigraph
    for phase in ['train', 'dev', 'test']:
        if phase != 'test' and args.fold:
            fea_filename = os.path.join(args.data, 'fold{}'.format(args.fold), '{}.fea'.format(phase))
            pos_filename = os.path.join(args.data, 'fold{}'.format(args.fold), '{}.pos'.format(phase))
        else:
            fea_filename = os.path.join(args.data, '{}.fea'.format(phase))
            pos_filename = os.path.join(args.data, '{}.pos'.format(phase))
        fea_file = open(fea_filename, 'rb')
        with open(pos_filename, 'r') as f:
            positions = [int(v.strip()) for v in f]
        dataset = GraphDataset(fea_file, positions)
        if args.multi_gpu and phase == 'train':
            sampler = t.utils.data.distributed.DistributedSampler(dataset)
            dataloader = t.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                                collate_fn=collate_fn, sampler=sampler, num_workers=1)
        else:
            dataloader = t.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                                shuffle=(phase=='train'), collate_fn=collate_fn, num_workers=1)
        dataloaders[phase] = dataloader
        datasets[phase] = dataset

    total_steps = int(args.epoch * len(datasets['train']) / args.batch_size)
    if model_config.name.find("BERT") != -1:
        if model_config.freeze:
            for param in model.bert4pretrain.parameters():
                param.requires_grad = False
        optimizer = getattr(optim, optimizer_config.name)(model.parameters(), **optimizer_config.values)
    else:
        optimizer = getattr(optim, optimizer_config.name)(model.parameters(), **optimizer_config.values)
    scheduler = getattr(optim.lr_scheduler, scheduler_config.name)(optimizer, **scheduler_config.values)

    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
    ckpt_file = os.path.join(args.save_dir, 'model.epoch{}.pt.tar'.format(args.conti))
    if args.conti is None:
        conti = 1
    elif os.path.isfile(ckpt_file) and args.conti is not None:
        load_ckpt(ckpt_file, model, optimizer, scheduler)
        conti = args.conti + 1
    else:
        raise Exception("No such path {}".format(ckpt_file))

    if args.multi_gpu:
        t.cuda.set_device(args.local_rank)
        model = model.cuda()
        t.distributed.init_process_group(backend='nccl', init_method='env://')
        model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[args.local_rank],
                                                output_device=args.local_rank)
    elif args.cuda:
        model = model.cuda()

    # pdb.set_trace()
    if args.log:
        writer = SummaryWriter(os.path.join(args.save_dir, 'logs'))
    else:
        writer = None
    pre_fn, step_fn, post_fn = tm.acc_metric_builder(args, scheduler_config, model,
                                                        optimizer, scheduler, writer, Log)

    phases = ['train', 'dev']
    if args.do_test:
        phases.append('test')
    for epoch in range(conti, conti+args.epoch):
        for phase in phases:
            pre_fn()
            if phase == 'train':
                if sampler is not None:
                    sampler.set_epoch(epoch)
                model.train()
            else:
                model.eval()

            pbar = tqdm(dataloaders[phase], position=args.local_rank)
            pbar.set_description("[{} Epoch {}]".format(phase, epoch))
            for data in pbar:
                optimizer.zero_grad()

                with t.set_grad_enabled(phase == 'train'):
                    result, loss = infer(data, model, args.cuda)
                    if phase == 'train':
                        loss.backward()
                        # t.nn.utils.clip_grad_norm_(model.parameters(), 7)
                        optimizer.step()
                step_fn(result, loss, pbar, phase)
            post_fn(phase, epoch)
    if args.log:
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', dest="cuda", action="store_true")
    parser.set_defaults(cuda=False)
    parser.add_argument('--log', dest='log', action='store_true', help='whether use tensorboard')
    parser.add_argument('--data', type=str, default="./inputs/train", help="input/target data name")
    parser.add_argument('--save_dir', type=str, default='./outputs/', help="model directory path")
    parser.add_argument('--config', type=str, default='model_config.json', help="config file")
    parser.add_argument('--batch_size', type=int, default=16, help="batch size")
    parser.add_argument('--epoch', type=int, default=20, help="number of epochs")
    parser.add_argument('--conti', type=int, default=None, help="the start epoch for continue training")
    parser.add_argument('--multi_gpu', dest='multi_gpu', action='store_true', help="use multi gpu")
    parser.add_argument('--fold', type=str, default='')
    parser.add_argument('--do_test', dest='do_test', action='store_true')
    parser.add_argument('--seed', type=int, default=2019)
    parser.set_defaults(multi_gpu=False, log=False, do_test=False)
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()
    t.manual_seed(args.seed)
    t.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    train(args)
