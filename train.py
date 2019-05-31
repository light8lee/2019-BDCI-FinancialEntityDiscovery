
import models
from proj_utils.files import save_ckpt, load_ckpt, load_config_from_json
from proj_utils.configuration import Config
from proj_utils.logs import log_info
from datasets import GraphDataset, collect_multigraph
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


def infer(data, model, criterion, seq_len, cuda):
    features, targets = data
    batch_ids, batch_masks, batch_laps = features

    if cuda:
        batch_ids = batch_ids.cuda()
        batch_masks = batch_masks.cuda()
        batch_laps = batch_laps.cuda()
    log_pred = model(batch_ids, batch_masks, batch_laps)
    loss = criterion(log_pred, targets)
    predictions = log_pred.argmax(1).numpy()
    labels = targets.numpy()
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    result = Counter({
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
        'total': tn+fp+fn+tp
    })

    return result, loss


def train(args):
    Log = log_info(os.path.join(args.save_dir, 'process.info'))
    Log(args)
    model_config, optimizer_config = Config.from_json(args.config)
    model_name = model_config.name
    model_class = getattr(models, model_name)

    if model_config.init_weight_path is None:
        model_config.init_weight = None
    else:
        model_config.init_weight = t.from_numpy(pickle.load(open(model_config.init_weight_path, 'rb')))

    if model_config.activation == 'identical':
        model_config.activation = lambda v: v
    elif model_config.activation == 'gelu':
        model_config.activation = models.layers.gelu
    else:
        model_config.activation = getattr(t, model_config.activation, None) or getattr(F, model_config.activation, None)

    model = model_class(**model_config.values)
    if args.multi_gpu:
        t.cuda.set_device(args.local_rank)
        model = model.cuda()
        t.distributed.init_process_group(backend='nccl', init_method='env://')
        model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[args.local_rank],
                                                output_device=args.local_rank)
    elif args.cuda:
        model = model.cuda()

    criterion = nn.NLLLoss()

    dataloaders = {}
    datasets = {}
    sampler = None
    for phase in ['train', 'dev', 'test']:
        fea_file = open(os.path.join(args.data, '{}.fea'.format(phase)), 'rb')
        tgt_filename = open(os.path.join(args.data, '{}.tgt'.format(phase)), 'rb')
        pos_filename = open(os.path.join(args.data, '{}.pos'.format(phase)), 'rb')
        with open(tgt_filename, 'r') as f:
            targets = [int(v.strip()) for v in f]
        with open(pos_filename, 'r') as f:
            positions = [int(v.strip()) for v in f]
        dataset = GraphDataset(fea_file, targets, positions)
        if args.multi_gpu and phase == 'train':
            sampler = t.utils.data.distributed.DistributedSampler(dataset)
            dataloader = t.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                                collate_fn=collect_multigraph, sampler=sampler, num_workers=1)
        else:
            dataloader = t.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                                shuffle=(phase=='train'), collate_fn=collect_multigraph, num_workers=1)
        dataloaders[phase] = dataloader
        datasets[phase] = dataset

    total_steps = int(args.epoch * len(datasets['train']) / args.batch_size)
    if args.multi_gpu:
        total_steps  = total_steps // t.distributed.get_world_size()
    optimizer = getattr(optim, optimizer_config.name)(model.parameters(), **optimizer_config.values)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=1, cooldown=0, min_lr=1e-5)  # TODO
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
    ckpt_file = os.path.join(args.save_dir, 'model.epoch{}.pt.tar'.format(args.conti))
    if args.conti is None:
        conti = 1
    elif os.path.isfile(ckpt_file) and args.conti is not None:
        load_ckpt(ckpt_file, model, optimizer)
        conti = args.conti + 1
    else:
        raise Exception("No such path {}".format(ckpt_file))

    epoch_loss = 10000
    best_f1 = 0
    for epoch in range(conti, conti+args.epoch):
        for phase in ['train', 'dev', 'test']:
            if phase == 'train':
                if sampler is not None:
                    sampler.set_epoch(epoch)
                model.train()
            else:
                model.eval()
            running_loss = 0.
            running_results = Counter()

            pbar = tqdm(dataloader[phase], position=args.local_rank)
            pbar.set_description("[{} Epoch {}]".format(phase, epoch))
            for data in pbar:
                optimizer.zero_grad()

                with t.set_grad_enabled(phase == 'train'):
                    result, loss = infer(data, model, criterion, model_config.seq_len, args.cuda)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_results += result
                running_loss += loss.item()

                if phase == 'train':
                    curr_lr = optimizer.param_groups[0]['lr']
                    if args.multi_gpu:
                        pbar.set_postfix(rank=args.local_rank, mean_loss=running_loss/running_results['total'], mean_acc=(running_results['tp']+running_results['tn'])/running_results['total'], lr=curr_lr)
                    else:
                        pbar.set_postfix(mean_loss=running_loss/running_results['total'], mean_acc=(running_results['tp']+running_results['tn'])/running_results['total'], lr=curr_lr)
                elif phase == 'dev':
                    pbar.set_postfix(mean_loss=running_loss/running_results['total'], mean_acc=running_results['tp']/running_results['total'])
            epoch_loss = running_loss / running_results['total']
            epoch_precision = Precision(running_results['tp'], running_results['fp'])
            epoch_recall = Recall(running_results['tp'], running_results['fn'])
            epoch_acc = (running_results['tp'] + running_results['tn']) / running_results['total']
            epoch_f1 = F1(epoch_precision, epoch_recall)
            if phase == 'dev':
                if epoch_f1 > best_f1:
                    best_f1 = epoch_f1
                    if args.multi_gpu:
                        if args.local_rank == 0:
                            Log('Epoch {}: Saving Rank({}) New Record... Acc: {}, P: {}, R: {}, F1: {} Loss: {}'.format(
                                epoch, args.local_rank, epoch_acc, epoch_precision, epoch_recall, epoch_f1, epoch_loss))
                            save_ckpt(os.path.join(args.save_dir, 'model.rank{}.epoch{}.pt.tar'.format(args.local_rank, epoch)),
                                                epoch, model.module.state_dict(), optimizer.state_dict())
                            unit_embedding = model.module.unit_embedding.weight.data.cpu().numpy()
                            pickle.dump(unit_embedding, open(os.path.join(args.save_dir, 'embedding.rank{}.epoch{}.dat'.format(args.local_rank, epoch)), 'wb'))
                    else:
                        Log('Epoch {}: Saving New Record... Acc: {}, P: {}, R: {}, F1: {} Loss: {}'.format(
                            epoch, epoch_acc, epoch_precision, epoch_recall, epoch_f1, epoch_loss))
                        save_ckpt(os.path.join(args.save_dir, 'model.epoch{}.pt.tar'.format(epoch)),
                                               epoch, model.state_dict(), optimizer.state_dict())
                        unit_embedding = model.unit_embedding.weight.data.cpu().numpy()
                        pickle.dump(unit_embedding, open(os.path.join(args.save_dir, 'embedding.epoch{}.dat'.format(epoch)), 'wb'))

                else:
                    if args.multi_gpu:
                        Log('Epoch {}:  Rank({}) Not Improved. Acc: {}, P: {}, R: {}, F1: {} Loss: {}'.format(
                            epoch, args.local_rank, epoch_acc, epoch_precision, epoch_recall, epoch_f1, epoch_loss))
                    else:
                        Log('Epoch {}: Not Improved. Acc: {}, P: {}, R: {}, F1: {} Loss: {}'.format(
                            epoch, epoch_acc, epoch_precision, epoch_recall, epoch_f1, epoch_loss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', dest="cuda", action="store_true")
    parser.set_defaults(cuda=False)
    parser.add_argument('--name', type=str, default="e4g", help="model name")
    parser.add_argument('--data', type=str, default="./inputs/train", help="input/target data name")
    parser.add_argument('--save_dir', type=str, default='./outputs/', help="model directory path")
    parser.add_argument('--config', type=str, default='model_config.json', help="config file")
    parser.add_argument('--batch_size', type=int, default=16, help="batch size")
    parser.add_argument('--epoch', type=int, default=20, help="number of epochs")
    parser.add_argument('--conti', type=int, default=None, help="the start epoch for continue training")
    parser.add_argument('--multi_gpu', dest='multi_gpu', action='store_true', help="use multi gpu")
    parser.set_defaults(multi_gpu=False)
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()
    train(args)
