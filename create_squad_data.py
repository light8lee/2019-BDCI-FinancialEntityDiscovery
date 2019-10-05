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
args = parser.parse_args()
random.seed(2019)
MAX_SEQ_LEN = 500


train_data = pd.read_csv('./data/Train_Data.csv', sep=',', dtype=str, encoding='utf-8')
test_data = pd.read_csv('./data/Test_Data.csv', sep=',', dtype=str, encoding='utf-8')

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
    for entity in entities:
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
                sub_texts.append(text[:MAX_SEQ_LEN])
                comma_pos = text.find('，', MAX_SEQ_LEN*3//4)
                if comma_pos == -1:
                    comma_pos = MAX_SEQ_LEN*3//4
                text = text[comma_pos:]
            else:
                sub_texts.append(text)
            print(sub_texts)

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
                        "question": "有哪些金融实体或公司？",
                        "id": '{}-{}'.format(idx, i)
                    }
                    qas.append(qa)
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
