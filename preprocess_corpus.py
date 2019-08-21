# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Create masked LM/next sentence masked_lm TF examples for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tokenization
import itertools
from tqdm import tqdm
import random
# import tokenization
import numpy as np
import pickle
import os
import argparse

flags = argparse.ArgumentParser()
flags.add_argument("input_dir")
flags.add_argument("output_dir")
flags.add_argument("vocab_file")
flags.add_argument('--type', default='kfold', type=str, choices=['kfold', 'origin'])
flags.add_argument('--do_lower_case', action='store_true', dest='do_lower_case')
flags.add_argument('--bert_style', action='store_true', dest='bert_style')
flags.set_defaults(do_lower_case=False, bert_style=False)
flags.add_argument('--max_seq_length', default=50, type=int)
flags.add_argument('--random_seed', type=int, default=12345)

FLAGS = flags.parse_args()
print(FLAGS)


def get_padded_tokens(tokens, tokenizer, max_seq_length, pad='after'):
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    if FLAGS.bert_style:
        max_seq_length += 3
    assert len(input_ids) <= max_seq_length

    to_pad = [0] * (max_seq_length - len(input_ids))
    if pad == 'before':
        input_ids = to_pad + input_ids
        input_mask = to_pad + input_mask
    elif pad == 'after':
        input_ids = input_ids + to_pad
        input_mask = input_mask + to_pad

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    return input_ids, input_mask


def get_pos(s, offset=0):
    pos_mat = collections.defaultdict(list)
    for i, c in enumerate(s):
        pos_mat[c].append(i+offset)
    return pos_mat


# def create_adj_from_tokens(instance, max_seq_length):
#     outer_positions = []
#     for i in range(len(instance.tokens_a)):
#         for j in range(len(instance.tokens_b)):
#             if instance.tokens_a[i] != instance.tokens_b[j]:
#                 outer_positions.append((i, j+max_seq_length))
#     return outer_positions


def write_instance_to_example_files(instances, tokenizer, max_seq_length, output_file, rng):
    tgt_writer = open('{}.tgt'.format(output_file), 'w')
    fea_writer = open('{}.fea'.format(output_file), 'wb')
    fea_pos_writer = open('{}.pos'.format(output_file), 'w')

    print('writing...')
    pbar = tqdm(enumerate(instances))
    pbar.set_description("[writing to files]")

    fea_pos = 0
    for (inst_index, instance) in pbar:
        if not instance.tokens_a or not instance.tokens_b:
            continue

        feature = collections.OrderedDict()
        target = collections.OrderedDict()

        if FLAGS.bert_style:
            inputs = ['[CLS]']
            inputs.extend(instance.tokens_a)
            inputs.append('[SEP]')
            ea_size = len(inputs)
            token_types = [0] * ea_size
            inputs.extend(instance.tokens_b)
            inputs.append('[SEP]')
            eb_size = len(inputs) - ea_size
            token_types += [1] * eb_size
            inputs, input_masks = get_padded_tokens(inputs, tokenizer, 2*max_seq_length)
            token_types += [0] * (len(inputs) - ea_size - eb_size)
            feature["inputs"] = inputs
            feature["input_masks"] = input_masks
            feature['token_types'] = token_types
        else:
            inputs_a, input_mask_a = get_padded_tokens(instance.tokens_a, tokenizer, max_seq_length)
            inputs_b, input_mask_b = get_padded_tokens(instance.tokens_b, tokenizer, max_seq_length)
            feature["inputs_a"] = inputs_a  # 输入ids
            feature["input_mask_a"] = input_mask_a  # mask ids的padding部分
            feature["inputs_b"] = inputs_b  # 输入ids
            feature["input_mask_b"] = input_mask_b  # mask ids的padding部分

        feature = tuple(feature.values())
        feature = pickle.dumps(feature)

        sz = fea_writer.write(feature)
        tgt_writer.write('{}\n'.format(instance.label))
        fea_pos_writer.write('{}\n'.format(fea_pos))
        fea_pos += sz
    tgt_writer.close()
    fea_writer.close()
    fea_pos_writer.close()



def create_training_instances(input_file, tokenizer, max_seq_length, rng):
    """Create `TrainingInstance`s from raw text."""

    # Input file format:
    with open(input_file, "r") as reader:
        for line in reader:
            line = line.strip()
            if not line:
                continue
            _, line_a, line_b, label = line.split('##')
            label = float(label)

            if FLAGS.bert_style:
                tokens_a = tokenizer.tokenize(line_a.replace('@@', ' '))
                tokens_b = tokenizer.tokenize(line_b.replace('@@', ' '))
            else:
                tokens_a = line_a.split('@@')
                tokens_b = line_b.split('@@')
            if not tokens_a or not tokens_b:
                continue
            yield create_instance(tokens_a, tokens_b, label, max_seq_length, rng)


def create_instance(tokens_a, tokens_b, label, max_seq_length, rng):
    """Creates `TrainingInstance`s for a single document."""

    # Account for [CLS] [SEP] (not used)
    max_num_tokens = max_seq_length
    truncate_seq(tokens_a, max_num_tokens, rng)
    truncate_seq(tokens_b, max_num_tokens, rng)
    instance = Instance(tokens_a, tokens_b, label)
    return instance


Instance = collections.namedtuple("Instance",
                                  ["tokens_a", "tokens_b", "label"])


def truncate_seq(tokens, max_num_tokens, rng):
    """Truncates a pair of sequences to a maximum sequence length."""
    while len(tokens) > max_num_tokens:
        tokens.pop()
    # while True:
    #     total_length = len(tokens)
    #     if total_length <= max_num_tokens:
    #         break

    #     if rng.random() < 0.5:
    #         del tokens[0]
    #     else:
    #         tokens.pop()

def _prepare(tokenizer, input_file, output_file):
    print("*** Reading from input files ***")
    rng = random.Random(FLAGS.random_seed)
    gen_instances = create_training_instances(
        input_file, tokenizer, FLAGS.max_seq_length, rng)

    print("*** Writing to output files ***")
    write_instance_to_example_files(gen_instances, tokenizer, FLAGS.max_seq_length,
                                    output_file, rng)


class Tokenizer(object):
    def __init__(self, vocab_file, do_lower_case, unk='[UNK]'):
        with open(vocab_file, encoding='utf-8') as f:
            self.id2char = list(map(str.strip, f))
            self.char2id = {ch: id for id, ch in enumerate(self.id2char)}
        self.unk = unk
        self.do_lower_case = do_lower_case
    
    def convert_tokens_to_ids(self, tokens):
        token_ids = []
        for token in tokens:
            if self.do_lower_case:
                token = token.lower()
            token_ids.append(
                self.char2id[token] if token in self.char2id else self.char2id[self.unk]
            )
        return token_ids
    
    def tokenize(self, tokens):
        return list(tokens)


def prepare_origin():
    if FLAGS.bert_style:
        tokenizer = tokenization.FullTokenizer(FLAGS.vocab_file, FLAGS.do_lower_case)
    else:
        tokenizer = Tokenizer(FLAGS.vocab_file, FLAGS.do_lower_case)

    for name in ['train', 'test', 'dev']:
        input_file = os.path.join(FLAGS.input_dir, '{}.txt'.format(name))
        output_file = os.path.join(FLAGS.output_dir, name)
        _prepare(tokenizer, input_file, output_file)


def prepare_kfold():
    if FLAGS.bert_style:
        tokenizer = tokenization.FullTokenizer(FLAGS.vocab_file, FLAGS.do_lower_case)
    else:
        tokenizer = Tokenizer(FLAGS.vocab_file, FLAGS.do_lower_case)

    for k in range(10):
        output_dir = os.path.join(FLAGS.output_dir, 'fold{}'.format(k))
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for name in ['train', 'dev']:
            input_file = os.path.join(FLAGS.input_dir, 'fold{}'.format(k), '{}.txt'.format(name))
            output_file = os.path.join(output_dir, name)
            _prepare(tokenizer, input_file, output_file)
    input_file = os.path.join(FLAGS.input_dir, 'test.txt')
    output_file = os.path.join(FLAGS.output_dir, 'test')
    _prepare(tokenizer, input_file, output_file)


if __name__ == "__main__":

    if FLAGS.type == 'origin':
        prepare_origin()
    elif FLAGS.type == 'kfold':
        prepare_kfold()
