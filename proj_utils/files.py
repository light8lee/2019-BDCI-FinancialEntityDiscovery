# -*- coding: utf-8 -*-
from .strings import q2b
from .strings import is_enword
from .strings import has_number
import os
import re
# import jieba
import json
import re
import numpy as np
import pickle
from collections import Counter


def get_chunks(f, drop_edge=False):
    data = []
    while True:
        try:
            chunk = pickle.load(f)
        except EOFError:
            break
        if drop_edge:
            chunk = (chunk[0], chunk[1], None)
        data.append(chunk)
    return data


def save_ckpt(ckpt_path, epoch, model_state, optimizer_state=None, scheduler_state=None):
    import torch
    ckpt_dict = {
        'epoch': epoch,
        'model_state': model_state,
        'optimizer_state': optimizer_state,
        'scheduler_state': scheduler_state,
    }
    torch.save(ckpt_dict, ckpt_path)


def load_ckpt(ckpt_path, model, optimizer=None, scheduler=None, cuda=False):
    import torch
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model_state'])
    if cuda:
        model = model.cuda()
    if optimizer and checkpoint['optimizer_state']:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    if scheduler and checkpoint['scheduler_state']:
        scheduler.load_state_dict(checkpoint['scheduler_state'])
    return checkpoint['optimizer_state']['param_groups'][0]['lr']


def cut_sent(para):
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    return para


def load_config_from_json(json_filename):
    with open(json_filename, 'r', encoding='utf-8') as f:
        v = f.read()
        obj = json.loads(v)
        model_config = obj.get('model', None)
        optimizer_config = obj.get('optimizer', None)
        scheduler_config = obj.get('scheduler', None)
        return model_config, optimizer_config, scheduler_config


def create_embedding(additional_words, embedding_filename, embedding_dim=300):
    vocabs = []
    assert additional_words[0] == '[PAD]'
    vocabs.extend(additional_words)
    embeddings = np.random.uniform(-1/embedding_dim, 1/embedding_dim, (len(additional_words), embedding_dim))
    embeddings[0] = 0
    pretrained_embeddings = []
    with open(embedding_filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cn_word, *values = line.split('\t')
            assert len(values) == embedding_dim
            values = [float(value) for value in values]
            pretrained_embeddings.append(values)
            vocabs.append(cn_word)
    pretrained_embeddings = np.array(pretrained_embeddings)
    embeddings = np.append(embeddings, pretrained_embeddings, axis=0)
    return embeddings, vocabs
