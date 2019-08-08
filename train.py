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
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix
from collections import Counter
import pdb

Precision = lambda tp, fp: tp / (tp + fp)
Recall = lambda tp, fn: tp / (tp + fn)
F1 = lambda p, r: ((2 * p * r) / (p + r)) if (p != 0) and (r != 0) else 0


def infer(data, model, criterion, cuda):
    features, targets = data
    batch_ids, batch_masks = features
    labels = targets.numpy()

    if cuda:
        if isinstance(batch_ids, t.Tensor):
            batch_ids = batch_ids.cuda()
            batch_masks = batch_masks.cuda()
        else:
            batch_ids = [v.cuda() for v in batch_ids]
            batch_masks = [v.cuda() for v in batch_masks]
        targets = targets.cuda()
    log_pred = model(batch_ids, batch_masks)
    loss = criterion(log_pred, targets)
    predictions = log_pred.argmax(1).cpu().numpy()
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    result = Counter({
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
        'total': tn+fp+fn+tp
    })

    return result, loss


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

    criterion = nn.NLLLoss()

    dataloaders = {}
    datasets = {}
    sampler = None
    for phase in ['train', 'dev', 'test']:
        if phase != 'test' and args.fold:
            fea_filename = os.path.join(args.data, 'fold{}'.format(args.fold), '{}.fea'.format(phase))
            tgt_filename = os.path.join(args.data, 'fold{}'.format(args.fold), '{}.tgt'.format(phase))
            pos_filename = os.path.join(args.data, 'fold{}'.format(args.fold), '{}.pos'.format(phase))
        else:
            fea_filename = os.path.join(args.data, '{}.fea'.format(phase))
            tgt_filename = os.path.join(args.data, '{}.tgt'.format(phase))
            pos_filename = os.path.join(args.data, '{}.pos'.format(phase))
        fea_file = open(fea_filename, 'rb')
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

    epoch_loss = 10000
    best_f1 = 0
    # pdb.set_trace()
    if args.log:
        writer = SummaryWriter(os.path.join(args.save_dir, 'logs'))
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

            pbar = tqdm(dataloaders[phase], position=args.local_rank)
            pbar.set_description("[{} Epoch {}]".format(phase, epoch))
            for data in pbar:
                optimizer.zero_grad()

                with t.set_grad_enabled(phase == 'train'):
                    result, loss = infer(data, model, criterion, args.cuda)
                    if phase == 'train':
                        loss.backward()
                        # t.nn.utils.clip_grad_norm_(model.parameters(), 7)
                        optimizer.step()
                running_results += result
                running_loss += loss.item()

                if phase == 'train':
                    curr_lr = optimizer.param_groups[0]['lr']
                    if args.multi_gpu:
                        pbar.set_postfix(rank=args.local_rank, mean_loss=running_loss/running_results['total'], mean_acc=(running_results['tp']+running_results['tn'])/running_results['total'], lr=curr_lr)
                    else:
                        pbar.set_postfix(mean_loss=running_loss/running_results['total'], mean_acc=(running_results['tp']+running_results['tn'])/running_results['total'], lr=curr_lr)
                else:
                    pbar.set_postfix(mean_loss=running_loss/running_results['total'], mean_acc=(running_results['tp']+running_results['tn'])/running_results['total'])
            epoch_loss = running_loss / running_results['total']
            epoch_precision = Precision(running_results['tp'], running_results['fp'])
            epoch_recall = Recall(running_results['tp'], running_results['fn'])
            epoch_acc = (running_results['tp'] + running_results['tn']) / running_results['total']
            epoch_f1 = F1(epoch_precision, epoch_recall)
            if args.log:
                writer.add_scalars('loss', {
                    phase: epoch_loss,
                }, epoch)
                writer.add_scalars('acc', {
                    phase: epoch_acc,
                }, epoch)
                writer.add_scalars('f1', {
                    phase: epoch_f1,
                }, epoch)
            if phase == 'dev':
                if scheduler_config.name == 'ReduceLROnPlateau':
                    scheduler.step(epoch_loss)
                else:
                    scheduler.step()
                if epoch_f1 > best_f1:
                    best_f1 = epoch_f1
                    if args.multi_gpu:
                        if args.local_rank == 0:
                            Log('dev Epoch {}: Saving Rank({}) New Record... Acc: {}, P: {}, R: {}, F1: {} Loss: {}'.format(
                                epoch, args.local_rank, epoch_acc, epoch_precision, epoch_recall, epoch_f1, epoch_loss))
                            save_ckpt(os.path.join(args.save_dir, 'model{}.rank{}.epoch{}.pt.tar'.format(args.fold, args.local_rank, epoch)),
                                                epoch, model.module.state_dict(), optimizer.state_dict(), scheduler.state_dict())
                            save_ckpt(os.path.join(args.save_dir, 'model{}.rank{}.best.pt.tar'.format(args.fold, args.local_rank)),
                                                epoch, model.module.state_dict(), optimizer.state_dict(), scheduler.state_dict())
                    else:
                        Log('dev Epoch {}: Saving New Record... Acc: {}, P: {}, R: {}, F1: {} Loss: {}'.format(
                            epoch, epoch_acc, epoch_precision, epoch_recall, epoch_f1, epoch_loss))
                        save_ckpt(os.path.join(args.save_dir, 'model{}.epoch{}.pt.tar'.format(args.fold, epoch)),
                                               epoch, model.state_dict(), optimizer.state_dict(), scheduler.state_dict())
                        save_ckpt(os.path.join(args.save_dir, 'model{}.best.pt.tar'.format(args.fold)),
                                               epoch, model.state_dict(), optimizer.state_dict(), scheduler.state_dict())
                else:
                    if args.multi_gpu:
                        Log('dev Epoch {}:  Rank({}) Not Improved. Acc: {}, P: {}, R: {}, F1: {} Loss: {}'.format(
                            epoch, args.local_rank, epoch_acc, epoch_precision, epoch_recall, epoch_f1, epoch_loss))
                    else:
                        Log('dev Epoch {}: Not Improved. Acc: {}, P: {}, R: {}, F1: {} Loss: {}'.format(
                            epoch, epoch_acc, epoch_precision, epoch_recall, epoch_f1, epoch_loss))
            else:
                Log('{} Epoch {}: Acc: {}, P: {}, R: {}, F1: {} Loss: {}'.format(
                    phase, epoch, epoch_acc, epoch_precision, epoch_recall, epoch_f1, epoch_loss))
    if args.log:
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', dest="cuda", action="store_true")
    parser.set_defaults(cuda=False)
    parser.add_argument('--name', type=str, default="e4g", help="model name")
    parser.add_argument('--log', dest='log', action='store_true', help='whether use tensorboard')
    parser.add_argument('--data', type=str, default="./inputs/train", help="input/target data name")
    parser.add_argument('--save_dir', type=str, default='./outputs/', help="model directory path")
    parser.add_argument('--config', type=str, default='model_config.json', help="config file")
    parser.add_argument('--batch_size', type=int, default=16, help="batch size")
    parser.add_argument('--epoch', type=int, default=20, help="number of epochs")
    parser.add_argument('--conti', type=int, default=None, help="the start epoch for continue training")
    parser.add_argument('--multi_gpu', dest='multi_gpu', action='store_true', help="use multi gpu")
    parser.add_argument('--fold', type=str, default='')
    parser.set_defaults(multi_gpu=False, log=False)
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()
    train(args)
