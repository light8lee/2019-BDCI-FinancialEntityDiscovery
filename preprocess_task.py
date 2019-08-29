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

flags = argparse.ArgumentParser()
flags.add_argument("input_dir")
flags.add_argument("output_dir")
flags.add_argument("vocab_file")
flags.add_argument("task")
flags.add_argument('--do_lower_case', action='store_true', dest='do_lower_case')
flags.set_defaults(do_lower_case=False, bert_style=False)
flags.add_argument('--max_seq_length', default=50, type=int)
flags.add_argument('--random_seed', type=int, default=12345)

def get_padded_tokens(tokens, tokenizer, vocabs, max_seq_length, pad='after'):
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    assert len(input_ids) <= max_seq_length, "len:{}".format(len(input_ids))

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


def prepare_qnli(args, tokenizer, vocabs, phase):
    output_file = os.path.join(args.output_dir, 'QNLI', phase)
    tgt_writer = open('{}.tgt'.format(output_file), 'w')
    fea_writer = open('{}.fea'.format(output_file), 'wb')
    fea_pos_writer = open('{}.pos'.format(output_file), 'w')

    input_file = os.path.join(args.input_dir, 'QNLI', '{}.tsv'.format(phase))
    datas = pd.read_csv(input_file, sep='\t', error_bad_lines=False,
                        warn_bad_lines=True, encoding='utf-8',
                        quoting=csv.QUOTE_NONE, dtype=str)
    datas.dropna(inplace=True)
    if phase == 'test':
        datas['label'] = 'not_entailment'
    fea_pos = 0
    for idx, line_a, line_b, label in zip(datas['index'], datas['question'], datas['sentence'], datas['label']):
        tokens_a = tokenizer.tokenize(line_a)
        tokens_b = tokenizer.tokenize(line_b)
        if not tokens_a or not tokens_b:
            continue
        tokens_a = tokens_a[:args.max_seq_length]
        tokens_b = tokens_b[:args.max_seq_length]
        inputs = ['[CLS]']
        inputs.extend(tokens_a)
        inputs.append('[SEP]')
        ea_size = len(inputs)
        token_types = [0] * ea_size

        inputs.extend(tokens_b)
        inputs.append('[SEP]')
        eb_size = len(inputs) - ea_size
        token_types += [1] * eb_size
        inputs, input_masks = get_padded_tokens(inputs, tokenizer, vocabs, 2*args.max_seq_length+3)
        token_types += [0] * (len(inputs) - ea_size - eb_size)

        feature = collections.OrderedDict()
        feature["index"] = idx
        feature["inputs"] = inputs
        feature["input_masks"] = input_masks
        feature['token_types'] = token_types
        feature = tuple(feature.values())
        feature = pickle.dumps(feature)

        sz = fea_writer.write(feature)
        label = 1 if label == 'entailment' else 0
        tgt_writer.write('{}\n'.format(label))
        fea_pos_writer.write('{}\n'.format(fea_pos))
        fea_pos += sz
    tgt_writer.close()
    fea_writer.close()
    fea_pos_writer.close()


def prepare_sts(args, tokenizer, vocabs, phase):
    output_file = os.path.join(args.output_dir, 'STS', phase)
    tgt_writer = open('{}.tgt'.format(output_file), 'w')
    fea_writer = open('{}.fea'.format(output_file), 'wb')
    fea_pos_writer = open('{}.pos'.format(output_file), 'w')

    input_file = os.path.join(args.input_dir, 'STS-B', '{}.tsv'.format(phase))
    datas = pd.read_csv(input_file, sep='\t', error_bad_lines=False,
                        warn_bad_lines=True, engine='python', encoding='utf-8',
                        quoting=csv.QUOTE_NONE, dtype=str)
    datas.dropna(inplace=True)
    if phase == 'test':
        datas['score'] = 0
    fea_pos = 0
    for idx, line_a, line_b, score in zip(datas['index'], datas['sentence1'], datas['sentence2'], datas['score']):
        tokens_a = tokenizer.tokenize(line_a)
        tokens_b = tokenizer.tokenize(line_b)
        if not tokens_a or not tokens_b:
            continue
        tokens_a = tokens_a[:args.max_seq_length]
        tokens_b = tokens_b[:args.max_seq_length]
        inputs = ['[CLS]']
        inputs.extend(tokens_a)
        inputs.append('[SEP]')
        ea_size = len(inputs)
        token_types = [0] * ea_size

        inputs.extend(tokens_b)
        inputs.append('[SEP]')
        eb_size = len(inputs) - ea_size
        token_types += [1] * eb_size
        inputs, input_masks = get_padded_tokens(inputs, tokenizer, vocabs, 2*args.max_seq_length+3)
        token_types += [0] * (len(inputs) - ea_size - eb_size)

        feature = collections.OrderedDict()
        feature["index"] = idx
        feature["inputs"] = inputs
        feature["input_masks"] = input_masks
        feature['token_types'] = token_types
        feature = tuple(feature.values())
        feature = pickle.dumps(feature)

        sz = fea_writer.write(feature)
        tgt_writer.write('{}\n'.format(score))
        fea_pos_writer.write('{}\n'.format(fea_pos))
        fea_pos += sz
    tgt_writer.close()
    fea_writer.close()
    fea_pos_writer.close()


def prepare_qqp(args, tokenizer, vocabs, phase):
    output_file = os.path.join(args.output_dir, 'QQP', phase)
    tgt_writer = open('{}.tgt'.format(output_file), 'w')
    fea_writer = open('{}.fea'.format(output_file), 'wb')
    fea_pos_writer = open('{}.pos'.format(output_file), 'w')

    input_file = os.path.join(args.input_dir, 'QQP', '{}.tsv'.format(phase))
    datas = pd.read_csv(input_file, sep='\t', error_bad_lines=False,
                        warn_bad_lines=True, engine='python', encoding='utf-8',
                        quoting=csv.QUOTE_NONE)
    datas.dropna(inplace=True)
    if phase == 'test':
        datas['is_duplicate'] = 0
    datas['is_duplicate'] = datas['is_duplicate'].astype(np.int)
    fea_pos = 0
    for idx, line_a, line_b, label in zip(datas['id'], datas['question1'], datas['question2'], datas['is_duplicate']):
        tokens_a = tokenizer.tokenize(line_a)
        tokens_b = tokenizer.tokenize(line_b)
        if not tokens_a or not tokens_b:
            continue
        tokens_a = tokens_a[:args.max_seq_length]
        tokens_b = tokens_b[:args.max_seq_length]
        inputs = ['[CLS]']
        inputs.extend(tokens_a)
        inputs.append('[SEP]')
        ea_size = len(inputs)
        token_types = [0] * ea_size

        inputs.extend(tokens_b)
        inputs.append('[SEP]')
        eb_size = len(inputs) - ea_size
        token_types += [1] * eb_size
        inputs, input_masks = get_padded_tokens(inputs, tokenizer, vocabs, 2*args.max_seq_length+3)
        token_types += [0] * (len(inputs) - ea_size - eb_size)

        feature = collections.OrderedDict()
        feature["index"] = idx
        feature["inputs"] = inputs
        feature["input_masks"] = input_masks
        feature['token_types'] = token_types
        feature = tuple(feature.values())
        feature = pickle.dumps(feature)

        sz = fea_writer.write(feature)
        tgt_writer.write('{}\n'.format(label))
        fea_pos_writer.write('{}\n'.format(fea_pos))
        fea_pos += sz
    tgt_writer.close()
    fea_writer.close()
    fea_pos_writer.close()


def main(args):
    vocabs = tokenization.load_vocab(args.vocab_file)
    tokenizer = tokenization.FullTokenizer(args.vocab_file, args.do_lower_case)

    for phase in ['test', 'train', 'dev']:
        print('phase:', phase)
        if args.task == 'QQP':
            prepare_qqp(args, tokenizer, vocabs, phase)
        elif args.task == 'STS':
            prepare_sts(args, tokenizer, vocabs, phase)
        elif args.task == 'QNLI':
            prepare_qnli(args, tokenizer, vocabs, phase)


if __name__ == '__main__':
    args = flags.parse_args()
    main(args)