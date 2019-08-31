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
flags.set_defaults(do_lower_case=False, bert_style=False)
flags.add_argument('--max_seq_length', default=50, type=int)
flags.add_argument('--random_seed', type=int, default=12345)


def get_padded_tokens(tokens, tags, vocabs, max_seq_length, pad='after'):
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
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                inputs = inputs[:args.max_seq_length]
                tags = tags[:args.max_seq_length]
                inputs.insert(0, '[CLS]')
                tags.insert(0, '[CLS]')
                inputs.append('[SEP]')
                tags.append('[SEP]')
                input_ids, input_masks, tag_ids = get_padded_tokens(inputs, tags, vocabs, args.max_seq_length+3)
                feature = collections.OrderedDict()
                feature["id"] = idx
                feature["inputs"] = input_ids
                feature["input_masks"] = input_masks
                feature["tags"] = tag_ids
                feature = tuple(feature.values())
                feature = pickle.dumps(feature)
                print(feature)

                sz = fea_writer.write(feature)
                fea_pos_writer.write('{}\n'.format(fea_pos))
                fea_pos += sz
                inputs = []
                tags = []
            elif line.find(' ') == -1:
                idx = line
            else:
                pair = line.split(' ')
                char = pair[0].lower()
                if char not in vocabs:
                    char = '[UNK]'
                inputs.append(char)
                tags.append(pair[1])
                if pair[1] == 'B':
                    num_entities += 1

    fea_writer.close()
    fea_pos_writer.close()


def main(args):
    vocabs = tokenization.load_vocab(args.vocab_file)

    for phase in ['test', 'train', 'dev']:
        print('phase:', phase)
        prepare_ner(args, vocabs, phase)


if __name__ == '__main__':
    args = flags.parse_args()
    main(args)
