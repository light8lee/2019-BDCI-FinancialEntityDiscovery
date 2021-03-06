import models
from sklearn.externals import joblib
from proj_utils.files import save_ckpt, load_ckpt, load_config_from_json
from proj_utils.configuration import Config
from proj_utils.logs import log_info
from dataset import MRCDataset, collect_mrc
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
from pytorch_transformers import optimization


SHOW = True
def infer(data, model, cuda, is_evaluate=False):
    idx, batch_ids, batch_masks, batch_begin_ids, batch_end_ids, batch_spans, \
        batch_inputs, lm_ids, batch_pairs = data
    global SHOW
    if SHOW:
        print(data)
        SHOW = False

    if cuda:
        batch_ids = batch_ids.cuda()
        batch_masks = batch_masks.cuda()
        batch_begin_ids = batch_begin_ids.cuda()
        batch_end_ids = batch_end_ids.cuda()
        batch_spans = batch_spans.cuda()
        # batch_flags = batch_flags.cuda()
        # batch_bounds = batch_bounds.cuda()
        # batch_extra = batch_extra.cuda()
        lm_ids = lm_ids.cuda()

    if is_evaluate:
        model_to_predict = model.module if hasattr(model, "module") else model
        with t.no_grad():
            predicts = model_to_predict.predict(batch_ids, batch_masks)
            #                                     flags=batch_flags, bounds=batch_bounds, extra=batch_extra)
        result = {
            'inputs': batch_inputs,
            'predict_pairs': predicts,
            'target_pairs': batch_pairs,
            'max_lens': [inputs.index('[SEP]') for inputs in batch_inputs],
            'batch_size': batch_masks.shape[0]
        }
        return result, 0

    loss = model(input_ids=batch_ids, input_masks=batch_masks,
                 target_begin_tag_ids=batch_begin_ids, target_end_tag_ids=batch_end_ids,
                 target_span_ids=batch_spans, lm_ids=lm_ids)
    result = {
        'batch_size': batch_masks.shape[0]
    }
    return result, loss


def train(args):
    Log = log_info(os.path.join(args.save_dir, 'process{}.info'.format(args.fold)))
    Log(args)
    model_config, optimizer_config, scheduler_config = Config.from_json(args.config)
    model_name = model_config.name
    model_class = getattr(models, model_name)
    Log(model_config.values)
    model = model_class(**model_config.values)

    dataloaders = {}
    datasets = {}
    sampler = None
    collate_fn = collect_mrc
    phases = ['train']
    if args.do_eval:
        phases.append('dev')
    if args.do_test:
        phases.append('test')
    for phase in phases:
        if phase != 'test' and args.fold:
            fea_filename = os.path.join(args.data, 'fold{}'.format(args.fold), '{}.fea'.format(phase))
            pos_filename = os.path.join(args.data, 'fold{}'.format(args.fold), '{}.pos'.format(phase))
        else:
            fea_filename = os.path.join(args.data, '{}.fea'.format(phase))
            pos_filename = os.path.join(args.data, '{}.pos'.format(phase))
        fea_file = open(fea_filename, 'rb')
        with open(pos_filename, 'r') as f:
            positions = [int(v.strip()) for v in f]
        dataset = MRCDataset(fea_file, positions)
        if args.multi_gpu and phase == 'train':
            sampler = t.utils.data.RandomSampler(dataset)
            dataloader = t.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                                collate_fn=collate_fn, sampler=sampler, num_workers=0)
        else:
            dataloader = t.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                                shuffle=(phase=='train'), collate_fn=collate_fn, num_workers=0)
        dataloaders[phase] = dataloader
        datasets[phase] = dataset

    if args.multi_gpu:
        args.n_gpu = t.cuda.device_count()
        model = model.cuda()
        model = t.nn.DataParallel(model)
    elif args.cuda:
        args.n_gpu = 1
        model = model.cuda()

    bert_parameters = list(map(id, model.bert4pretrain.parameters()))
    other_parameters = filter(lambda p: id(p) not in bert_parameters, model.parameters())
    if hasattr(optim, optimizer_config.name):
        optimizer = getattr(optim, optimizer_config.name)([
            {'params': other_parameters, 'lr': optimizer_config.lr*args.scale_rate},
            {'params': model.bert4pretrain.parameters()}
        ], **optimizer_config.values)
        scheduler = getattr(optim.lr_scheduler, scheduler_config.name)(optimizer, **scheduler_config.values)
    else:
        t_total = len(dataloaders['train']) * args.epoch
        # no_decay = ['bias', 'LayerNorm.weight']
        # optimizer_grouped_parameters = [
        #     {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        #     {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        #     ]
        optimizer = getattr(optimization, optimizer_config.name)([
            {'params': other_parameters, 'lr': optimizer_config.lr*args.scale_rate},
            {'params': model.bert4pretrain.parameters()}
        ], **optimizer_config.values)
        scheduler = getattr(optimization, scheduler_config.name)(optimizer, t_total=t_total, **scheduler_config.values)

    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)

    # pdb.set_trace()
    if args.log:
        writer = SummaryWriter(os.path.join(args.save_dir, 'logs'))
    else:
        writer = None
    pre_fn, step_fn, post_fn = tm.mrc_acc_metric_builder(args, scheduler_config, model,
                                                        optimizer, scheduler, writer, Log)

    for epoch in range(1, 1+args.epoch):
        for phase in phases:
            pre_fn()
            if phase == 'train':
                model.train()
            else:
                model.eval()

            pbar = tqdm(dataloaders[phase])
            pbar.set_description("[{} Epoch {}]".format(phase, epoch))
            for data in pbar:
                optimizer.zero_grad()

                with t.set_grad_enabled(phase == 'train'):
                    result, loss = infer(data, model, args.cuda, is_evaluate=phase!='train')
                    if args.multi_gpu and args.n_gpu > 1:
                        loss = loss.mean()
                    if phase == 'train':
                        loss.backward()
                        # t.nn.utils.clip_grad_norm_(model.parameters(), 7)
                        optimizer.step()
                step_fn(result, loss, pbar, phase)
            post_fn(phase, epoch)
    if args.log:
        writer.close()
    with open(os.path.join(args.save_dir, 'invalid_entities'), 'wb') as f:
        pickle.dump(tm.Invalid_entities, f)


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
    parser.add_argument('--scale_rate', type=float, default=1., help="scale rate of learning rate")
    parser.add_argument('--multi_gpu', dest='multi_gpu', action='store_true', help="use multi gpu")
    parser.add_argument('--fold', type=str, default='')
    parser.add_argument('--do_test', dest='do_test', action='store_true')
    parser.add_argument('--do_eval', dest='do_eval', action='store_true')
    parser.add_argument('--seed', type=int, default=2019)
    parser.set_defaults(multi_gpu=False, log=False, do_test=False, do_eval=False)
    args = parser.parse_args()
    t.manual_seed(args.seed)
    t.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    train(args)
