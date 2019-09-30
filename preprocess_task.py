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
from proj_utils import BIO_TAG2ID

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


def get_padded_tokens(tokens, tags, vocabs, max_seq_length, pad='after'):
    tokens = [token.lower() if token not in ['[CLS]', '[SEP]'] else token for token in tokens]
    tokens = [token if token in vocabs else '[UNK]' for token in tokens]
    input_ids = tokenization.convert_tokens_to_ids(vocabs, tokens)
    input_mask = [1] * len(input_ids)
    tag_ids = [BIO_TAG2ID[tag] for tag in tags]
    assert len(input_ids) <= max_seq_length, "len:{}".format(len(input_ids))

    to_pad = [0] * (max_seq_length - len(input_ids))
    if pad == 'before':
        input_ids = to_pad + input_ids
        input_mask = to_pad + input_mask
        tag_ids = to_pad + tag_ids
    elif pad == 'after':
        input_ids = input_ids + to_pad
        input_mask = input_mask + to_pad
        tag_ids = tag_ids + to_pad

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(tag_ids) == max_seq_length
    return input_ids, input_mask, tag_ids


def prepare_ner(args, vocabs, phase):
    output_file = os.path.join(args.output_dir, phase)
    fea_writer = open('{}.fea'.format(output_file), 'wb')
    fea_pos_writer = open('{}.pos'.format(output_file), 'w')

    input_file = os.path.join(args.input_dir, '{}.txt'.format(phase))

    fea_pos = 0
    num_entities = 0
    inputs = []
    tags = []
    idx = 0
    idxs = set()
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line[:-1]
            if line.startswith('$'*10):
                inputs = inputs[:args.max_seq_length]
                tags = tags[:args.max_seq_length]
                # print(inputs)
                # print(tags)
                inputs.insert(0, '[CLS]')
                tags.insert(0, '[CLS]')
                inputs.append('[SEP]')
                tags.append('[SEP]')
                input_ids, input_masks, tag_ids = get_padded_tokens(inputs, tags, vocabs, args.max_seq_length+2)
                feature = collections.OrderedDict()
                feature["id"] = idx
                feature["input_ids"] = input_ids
                feature["input_masks"] = input_masks
                feature["tags"] = tag_ids
                feature["inputs"] = inputs
                feature = tuple(feature.values())
                # print(feature)
                feature = pickle.dumps(feature)

                sz = fea_writer.write(feature)
                fea_pos_writer.write('{}\n'.format(fea_pos))
                fea_pos += sz
                inputs = []
                tags = []
            elif line.startswith('^'*10):
                idx = line.replace('^', '')
                idxs.add(idx)
            else:
                pair = line.split(' ')
                if not pair[0] or not pair[1]:
                    continue
                token, tag = pair
                inputs.append(token)
                tags.append(tag)
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
