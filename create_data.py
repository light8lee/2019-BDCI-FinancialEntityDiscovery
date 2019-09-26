# coding: utf-8

import random
import pandas as pd
import os
import numpy as np
import argparse
import re


random.seed(2019)
MAX_SEQ_LEN = 500

img = re.compile(r'\{IMG:\d{1,}\}')
img2 = re.compile(r'<!--(IMG[_\d\s]+)-->')
time = re.compile(r'(\d{4}-\d{2}-\d{2})|(\d{2}:\d{2}:\d{2})')
tag = re.compile(r'<(\d|[a-z".A-Z/]|\s)+>')
ques = re.compile(r'[?#/▲◆]+')
vx = re.compile(r'(v\d+)|(微信:\d+)')
user = re.compile(r'@.*:')
url = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
plain = re.compile(r'\s+')
dots = re.compile(r'([.。，!?？！．,，＼／、])+')
num = re.compile(r'\d+')
emoji = re.compile(r"[^\U00000000-\U0000d7ff\U0000e000-\U0000ffff]", flags=re.UNICODE)


def clean(text):
    text = text.replace('&nbsp;', '')
    text = url.sub('', text)
    text = emoji.sub('', text)
    text = plain.sub(' ', text)
    text = img.sub('', text)
    text = img2.sub('', text)
    text = time.sub('', text)
    text = tag.sub('', text)
    text = ques.sub('', text)
    text = dots.sub(r'\1', text)
    text = vx.sub('', text)
    text = user.sub('', text)
    text = num.sub('0', text)
    return text



def findall(text, entity):
    text_length = len(text)
    entity_length = len(entity)
    result = []
    begin = 0
    if not entity:
        return result
    while True:
        pos = text.find(entity, begin)
        if pos != -1:
            result.append((pos, pos+entity_length))
            begin += pos + entity_length
        else:
            return result


def create_tags(text, entities):
    tags = ['O'] * len(text)
    has_entity = False
    for entity in entities:
        # print('entity:', entity)
        for begin, end in findall(text, entity):
            tags[begin] = 'B'
            has_entity = True
            for i in range(begin+1, end):
                tags[i] = 'I'
    # print(tags)
    return tags, has_entity
        

def create_data(data, output_filename, is_test):
    line = 0
    with open(output_filename, 'w', encoding='utf-8') as f:
        for idx, text, title, entities in zip(data['id'], data['cleaned_text'], data['cleaned_title'], data['unknownEntities']):
            # print('---------------line:', line)
            entities = entities.split(';')
            sub_texts = []

            title = title.strip()
            text += title
            while len(text) > MAX_SEQ_LEN:
                sub_texts.append(text[:MAX_SEQ_LEN])
                text = text[MAX_SEQ_LEN:]
            else:
                sub_texts.append(text)
            # print(sub_texts)

            for sub_text in sub_texts:
                sub_text = sub_text.strip()
                if not sub_text:
                    continue
                if not is_test and len(sub_text) < 6:
                    continue
                tags, has_entity = create_tags(sub_text, entities)
                f.write('^'*10)
                f.write(idx)
                f.write('\n')
                print(sub_text)
                for char, tag in zip(sub_text, tags):
                    f.write('{} {}\n'.format(char, tag))
                f.write('$'*10)
                f.write('\n')
            line += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir')
    args = parser.parse_args()

    train_data = pd.read_csv('./data/Train_Data.csv', sep=',', dtype=str, encoding='utf-8')
    test_data = pd.read_csv('./data/Test_Data.csv', sep=',', dtype=str, encoding='utf-8')

    train_data.fillna('', inplace=True)
    test_data.fillna('', inplace=True)

    train_data['cleaned_text'] = train_data['text'].apply(clean)
    train_data['cleaned_title'] = train_data['title'].apply(clean)
    test_data['cleaned_text'] = test_data['text'].apply(clean)
    test_data['cleaned_title'] = test_data['title'].apply(clean)
    test_data['unknownEntities'] = ''

    train_data = train_data.sample(frac=1, random_state=2019).reset_index(drop=True)
    dev_data = train_data.tail(100)
    train_data = train_data.head(train_data.shape[0]-100)

    create_data(train_data, '{}/train.txt'.format(args.output_dir), False)

    create_data(dev_data, '{}/dev.txt'.format(args.output_dir), False)

    create_data(test_data, '{}/test.txt'.format(args.output_dir), True)