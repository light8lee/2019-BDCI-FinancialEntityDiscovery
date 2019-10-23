import collections
import tokenization
import itertools
from tqdm import tqdm
import random
import numpy as np
import pandas as pd
import pickle
import os
import csv
import argparse
import jieba.posseg as pseg
from proj_utils import BIO_TAG2ID, POS_FLAGS, POS_FLAGS_TO_IDS, WORD_BOUNDS_TO_IDS


flags = argparse.ArgumentParser()
flags.add_argument("input_dir")
flags.add_argument("output_dir")
flags.add_argument("vocab_file")
flags.add_argument('--do_lower_case', action='store_true', dest='do_lower_case')
flags.add_argument('--need_dev', action='store_true', dest='need_dev')
flags.add_argument('--no_test', action='store_true', dest='no_test')
flags.set_defaults(do_lower_case=False, need_dev=False, no_test=False)
flags.add_argument('--max_seq_length', default=50, type=int)
flags.add_argument('--random_seed', type=int, default=12345)
flags.add_argument('--rich_tag', default=False, action='store_true')
flags.add_argument('--duplicate', default=1, type=int)
flags.add_argument('--augment', default=False, action='store_true')


def tokens_to_flags(tokens):
    text = ''.join(tokens[1:-1])
    flags = ['[CLS]']
    bounds = ['[CLS]']
    for word, flag in pseg.cut(text):
        if flag not in POS_FLAGS:
            flag = 'un'
        if len(word) == 1:
            bounds.append('S')
        else:
            bounds.append('B')
            bounds.extend('I'*(len(word)-1))
        for ch in word:
            flags.append(flag)

    flags.append('[SEP]')
    bounds.append('[SEP]')
    return flags, bounds


def get_padded_tokens(tokens, tags, flags, bounds, extra_features, vocabs, max_seq_length, pad='after'):
    tokens = [token.lower() if token not in ['[CLS]', '[SEP]'] else token for token in tokens]
    tokens = [token if token in vocabs else '[UNK]' for token in tokens]
    input_ids = tokenization.convert_tokens_to_ids(vocabs, tokens)
    input_mask = [1] * len(input_ids)
    tag_ids = [BIO_TAG2ID[tag] for tag in tags]
    flag_ids = [POS_FLAGS_TO_IDS[flag] for flag in flags]
    bound_ids = [WORD_BOUNDS_TO_IDS[bound] for bound in bounds]
    assert len(input_ids) <= max_seq_length, "len:{}".format(len(input_ids))

    to_pad = [0] * (max_seq_length - len(input_ids))
    fea_to_pad = [[0] * len(extra_features[0])] * (max_seq_length - len(input_ids))
    if pad == 'before':
        input_ids = to_pad + input_ids
        input_mask = to_pad + input_mask
        tag_ids = to_pad + tag_ids
        flag_ids = to_pad + flag_ids
        bound_ids = to_pad + bound_ids
        extra_features = fea_to_pad + extra_features
    elif pad == 'after':
        input_ids = input_ids + to_pad
        input_mask = input_mask + to_pad
        tag_ids = tag_ids + to_pad
        flag_ids = flag_ids + to_pad
        bound_ids = bound_ids + to_pad
        extra_features = extra_features + fea_to_pad

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(tag_ids) == max_seq_length
    return input_ids, input_mask, tag_ids, flag_ids, bound_ids, extra_features


def do_augment(input_ids, vocabs):
    sep_id = vocabs['[SEP]']
    mask_id = vocabs['[MASK]']
    lm_ids = [-1] * len(input_ids)
    if random.random() < 0.6:
        return input_ids, lm_ids
    for i in range(1, len(input_ids)):
        if input_ids[i] == sep_id:
            break
        if random.random() < 0.5:
            lm_ids[i] = input_ids[i]
            if random.random() < 0.2:
                input_ids[i] = mask_id
            elif str.islower(input_ids[i]):
                input_ids[i] = random.randint(vocabs['a'], vocabs['z'])
            elif str.isnumeric(input_ids[i]):
                inputs_ids[i] = random.randint(vocabs['0'], vocabs['9'])
            elif 672 <= inputs_ids[i] <= 7993:
                input_ids[i] = random.randint(672, 7993)
    return input_ids, lm_ids


def prepare_ner(args, vocabs, phase):
    output_file = os.path.join(args.output_dir, phase)
    fea_writer = open('{}.fea'.format(output_file), 'wb')
    fea_pos_writer = open('{}.pos'.format(output_file), 'w')

    input_file = os.path.join(args.input_dir, '{}.txt'.format(phase))

    fea_pos = 0
    num_entities = 0
    inputs = []
    tags = []
    extra_features = []
    idx = 0
    idxs = set()
    duplicate = 1 if phase != 'train' else args.duplicate
    augment = False if phase != 'train' else args.augment
    for _ in range(duplicate):
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line[:-1]
                if line.startswith('$'*10):
                    inputs = inputs[:args.max_seq_length]
                    tags = tags[:args.max_seq_length]
                    # print(inputs)
                    # print(tags)
                    inputs.insert(0, '[CLS]')
                    if args.rich_tag:
                        tags.insert(0, '[CLS]')
                    else:
                        tags.insert(0, 'O')
                    extra_features.insert(0, [0]*len(extra_features[0]))
                    inputs.append('[SEP]')

                    if args.rich_tag:
                        tags.append('[SEP]')
                    else:
                        tags.append('O')
                    extra_features.append([0]*len(extra_features[0]))
                    flags, bounds = tokens_to_flags(inputs)
                    input_ids, input_masks, tag_ids, flag_ids, bound_ids, extra_features, = \
                        get_padded_tokens(inputs, tags, flags, bounds, extra_features, vocabs, args.max_seq_length+2)
                    if augment:
                        input_ids, lm_ids = do_augment(input_ids, vocabs)
                    else:
                        lm_ids = [-1] * len(input_ids)
                    feature = collections.OrderedDict()
                    feature["id"] = idx
                    feature["input_ids"] = input_ids
                    feature["input_masks"] = input_masks
                    feature["tags"] = tag_ids
                    feature["inputs"] = inputs
                    feature["flag_ids"] = flag_ids
                    feature["bound_ids"] = bound_ids
                    feature["extra"] = extra_features
                    feature["lm_ids"] = lm_ids
                    feature = tuple(feature.values())
                    # print(feature)
                    feature = pickle.dumps(feature)

                    sz = fea_writer.write(feature)
                    fea_pos_writer.write('{}\n'.format(fea_pos))
                    fea_pos += sz
                    inputs = []
                    tags = []
                    extra_features = []
                elif line.startswith('^'*10):
                    idx = line.replace('^', '')
                    idxs.add(idx)
                else:
                    pair = line.split(' ')
                    if not pair[0] or not pair[1]:
                        continue
                    token, tag, *extra_fea = pair
                    inputs.append(token)
                    tags.append(tag)
                    extra_features.append([float(v) for v in extra_fea])
                    if pair[1] == 'B':
                        num_entities += 1

    print('totoal entities:', num_entities)
    print('totoal lines:', len(idxs))
    fea_writer.close()
    fea_pos_writer.close()


def main(args):
    vocabs = tokenization.load_vocab(args.vocab_file)
    # tokenizer = tokenization.BasicTokenizer(do_lower_case=False)
    phases = ['train']
    if args.need_dev:
        phases.append('dev')
    if not args.no_test:
        phases.append('test')

    for phase in phases:
        print('phase:', phase)
        prepare_ner(args, vocabs, phase)


if __name__ == '__main__':
    args = flags.parse_args()
    main(args)
