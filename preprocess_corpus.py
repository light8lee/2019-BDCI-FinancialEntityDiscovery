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
import itertools
from tqdm import tqdm
import random
import tokenization
import numpy as np
import pickle
import os
import tensorflow as tf

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_dir", None,
                    "input file dir")

flags.DEFINE_string(
    "output_dir", None,
    "output file dir")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer("max_seq_length", 50, "Maximum sequence length.")

flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")

class TrainingInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(self, tokens, masked_lm_positions, masked_lm_labels):
        self.tokens = tokens
        # self.is_random_next = is_random_next
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels

    def __str__(self):
        s = ""
        s += "tokens: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.tokens]))
        s += "masked_lm_positions: %s\n" % (" ".join(
            [str(x) for x in self.masked_lm_positions]))
        s += "masked_lm_labels: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.masked_lm_labels]))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()

def get_padded_tokens(tokens, tokenizer, max_seq_length, pad='after'):
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
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


def create_adj_from_tokens(instance, max_seq_length):
    pos1 = get_pos(instance.tokens_a)
    pos2 = get_pos(instance.tokens_b, max_seq_length)
    up_positions = []
    for key in pos1:
        up_positions.extend(itertools.product(pos1[key], pos2[key]))
    rows, cols = zip(*up_positions)  #  上三角矩阵，只记录了tokens_a到tokens_b的边对应的邻接矩阵
    return rows, cols


def write_instance_to_example_files(instances, tokenizer, max_seq_length, output_file, rng):
    tgt_writer = open('{}.tgt'.format(output_file), 'w')
    fea_writer = open('{}.fea'.format(output_file), 'wb')
    fea_pos_writer = open('{}.pos'.format(output_file), 'w')

    print('writing...')
    pbar = tqdm(enumerate(instances))
    pbar.set_description("[writing to files]")

    fea_pos = 0
    for (inst_index, instance) in pbar:

        feature = collections.OrderedDict()
        target = collections.OrderedDict()

        inputs_a, input_mask_a = get_padded_tokens(instance.tokens_a, tokenizer, max_seq_length)
        inputs_b, input_mask_b = get_padded_tokens(instance.tokens_b, tokenizer, max_seq_length)
        feature["inputs_a"] = inputs_a  # 输入ids
        feature["input_mask_a"] = input_mask_a  # mask ids的padding部分
        feature["inputs_b"] = inputs_b  # 输入ids
        feature["input_mask_b"] = input_mask_b  # mask ids的padding部分
        feature['rows'], feature['cols'] = create_adj_from_tokens(instance, max_seq_length)

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
    vocab_words = list(tokenizer.vocab.keys())
    with open(input_file, "r") as reader:
        for line in reader:
            line = line.strip()
            if not line:
                continue
            line = ''.join(line.split('@@'))
            _, line_a, line_b, label = line.split('##')
            label = int(label)
            line_a = tokenization.convert_to_unicode(line_a)
            line_b = tokenization.convert_to_unicode(line_b)

            tokens_a = tokenizer.tokenize(line_a)
            tokens_b = tokenizer.tokenize(line_b)
            yield create_instance(tokens_a, tokens_b, label, vocab_words, max_seq_length, rng)


def create_instance(tokens_a, tokens_b, label, vocab_words, max_seq_length, rng):
    """Creates `TrainingInstance`s for a single document."""

    # Account for [CLS] [SEP]
    max_num_tokens = max_seq_length - 2
    truncate_seq(tokens_a, max_num_tokens, rng)
    truncate_seq(tokens_b, max_num_tokens, rng)
    tokens_a.insert(0, '[CLS]')
    tokens_b.insert(0, '[CLS]')
    tokens_a.append('[SEP]')
    tokens_b.append('[SEP]')
    instance = Instance(tokens_a, tokens_b, label)
    return instance


Instance = collections.namedtuple("Instance",
                                  ["tokens_a", "tokens_b", "label"])


def truncate_seq(tokens, max_num_tokens, rng):
    """Truncates a pair of sequences to a maximum sequence length."""
    while True:
        total_length = len(tokens)
        if total_length <= max_num_tokens:
            break

        if rng.random() < 0.5:
            del tokens[0]
        else:
            tokens.pop()


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    for name in ['train', 'test', 'dev']:
        tf.logging.info("*** Reading from input files ***")
        input_file = os.path.join(FLAGS.input_dir, '{}.txt'.format(name))
        rng = random.Random(FLAGS.random_seed)
        gen_instances = create_training_instances(
            input_file, tokenizer, FLAGS.max_seq_length, rng)

        tf.logging.info("*** Writing to output files ***")

        output_file = os.path.join(FLAGS.output_dir, name)
        write_instance_to_example_files(gen_instances, tokenizer, FLAGS.max_seq_length,
                                        output_file, rng)


if __name__ == "__main__":
    flags.mark_flag_as_required("input_dir")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("vocab_file")
    tf.app.run()
