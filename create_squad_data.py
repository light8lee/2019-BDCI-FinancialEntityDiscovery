# https://gitlab.com/snowhitiger/speakerextraction/blob/master/label_honglou.txt
# coding: utf-8


import random
import json
import pandas as pd
import os
import numpy as np
import re
import argparse
import create_data


parser = argparse.ArgumentParser()
parser.add_argument('output_dir')
parser.add_argument('--max_seq_len', type=int, default=400)
args = parser.parse_args()
random.seed(2019)
MAX_SEQ_LEN = args.max_seq_len


train_data = pd.read_csv('./round2_data/Train_Data.csv', sep=',', dtype=str, encoding='utf-8')
test_data = pd.read_csv('./round2_data/Test_Data.csv', sep=',', dtype=str, encoding='utf-8')

train_data.fillna('', inplace=True)
test_data.fillna('', inplace=True)

train_data['cleaned_text'] = train_data['text'].apply(create_data.clean)
train_data['cleaned_title'] = train_data['title'].apply(create_data.clean)
test_data['cleaned_text'] = test_data['text'].apply(create_data.clean)
test_data['cleaned_title'] = test_data['title'].apply(create_data.clean)
test_data['unknownEntities'] = ''

create_data.remove_chars(train_data, test_data)

train_data = train_data.sample(frac=1, random_state=2019).reset_index(drop=True)
dev_data = train_data.tail(100)
train_data = train_data.head(train_data.shape[0]-100)


def create_tags(text, entities):
    text = text.lower()
    for entity in entities:
        entity = entity.lower()
        start_pos = text.find(entity)
        if start_pos != -1:
            yield (start_pos, entity)


def create_squad_data(data, output_filename, is_test):
    line = 0
    datas = []
    i = 0
    with open(output_filename, 'w', encoding='utf-8') as f:
        for idx, text, entities in zip(data['id'], data['cleaned_text'], data['unknownEntities']):
            # print('---------------line:', line)
            entities = entities.split(';')
            sub_texts = []

            while len(text) > MAX_SEQ_LEN:
                right_bound = -1
                for stop in '，。？?':
                    tmp = text[:MAX_SEQ_LEN].rfind(stop, MAX_SEQ_LEN//2)
                    if tmp > right_bound:
                        right_bound = tmp
                if right_bound == -1:
                    right_bound = MAX_SEQ_LEN-1
                sub_texts.append(text[:right_bound+1])
                text = text[right_bound+1:]
            else:
                sub_texts.append(text)
            # print(sub_texts)
            # sub_texts.append(text)

            for sub_text in sub_texts:
                sub_text = sub_text.strip()
                # sub_text = sub_text.replace(' ', '※')
                if not sub_text:
                    continue
                qas = []
                for pos in create_tags(sub_text, entities):
                    answers = [{
                        "answer_start": pos[0],
                        "text": pos[1]
                    }]
                    qa = {
                        "answers": answers,
                        "question": "有哪些金融公司、平台、中心、投资、币、银行、基金、外汇、集团、链、股份、商城、店、资本、家园、金服、交易所、理财、贷款？",
                        "id": '{}-{}'.format(idx, i)
                    }
                    qas.append(qa)
                if not qas:
                    continue
                para_entry = dict()
                para_entry["context"] = sub_text
                para_entry["qas"] = qas
                data = {
                    "title": "金融实体",
                    "paragraphs": [para_entry]
                }
                datas.append(data)
                i += 1
    outputs = {
        "data": datas,
        "version": "chinese_squad_v1.0"
    }
    with open(output_filename, 'w') as f:
        f.write(json.dumps(outputs, ensure_ascii=False))


create_squad_data(train_data, '{}/train.json'.format(args.output_dir), False)


create_squad_data(dev_data, '{}/dev.json'.format(args.output_dir), True)


create_squad_data(test_data, '{}/test.json'.format(args.output_dir), True)
