# coding: utf-8

import random
import pandas as pd
import os
import numpy as np
import argparse
import re
from collections import defaultdict, Counter

random.seed(2019)

img = re.compile(r'\{IMG:\d{1,}\}')
img2 = re.compile(r'<!--(IMG[_\d\s]+)-->')
time = re.compile(r'(\d{4}-\d{2}-\d{2})|(\d{2}:\d{2}:\d{2})')
tag = re.compile(r'<(\d|[a-z".A-Z/]|\s)+>')
vx = re.compile(r'微信[:：]?[a-zA-Z0-9]+')
user = re.compile(r'@.*:')
url = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
plain = re.compile(r'[\s\t\n\b]+')
dots = re.compile(r'([.。，!?？！．,＼／、#])+')
num = re.compile(r'\d+')
emoji = re.compile(r"[^\U00000000-\U0000d7ff\U0000e000-\U0000ffff\u200B®▼☑▲◆]", flags=re.UNICODE)
extra_chars = set("!#$%&\()*+,-./:;<=>?@[\\]^_`{|}~！#￥%&？《》{}“”，：‘’。（）·、；【】")


def clean(text):
    text = text.replace('&nbsp;', '，')
    text = text.replace(',', '，')
    text = vx.sub('，', text)
    text = url.sub('，', text)
    text = emoji.sub('，', text)
    text = plain.sub('，', text)
    text = img.sub('，', text)
    text = img2.sub('，', text)
    text = time.sub('，', text)
    text = tag.sub('，', text)
    text = user.sub('，', text)
    # text = num.sub('0', text)

    text = text.replace("\xa0", "")
    text = text.replace("\b", "")
    text = text.replace('"', "")
    text = re.sub(r"\t|\n|\x0b|\x1c|\x1d|\x1e", "", text)
    text = text.strip()
    text = re.sub(r'\?\?+', '', text)
    text = re.sub(r'\{IMG:.?.?.?\}', '，', text)
    text = re.sub(r'\t|\n', '，', text)
    text = dots.sub(r'\1', text)
    return text


def remove_chars(train_df, test_df):
    test_df['cleaned_text'] = test_df['cleaned_title'] + '，' + test_df['cleaned_text']
    train_df['cleaned_text'] = train_df['cleaned_title']  + '，' + train_df['cleaned_text']
    additional_chars = set()
    for t in list(test_df['cleaned_text']) + list(train_df['cleaned_text']):
        additional_chars.update(re.findall(r'[^\u4e00-\u9fa5a-zA-Z0-9\*]', t))

    # 一些需要保留的符号
    additional_chars = additional_chars - extra_chars

    def remove_additional_chars(text):
        for x in additional_chars:
            text = text.replace(x, "")
        return text

    train_df["cleaned_text"] = train_df["cleaned_text"].apply(remove_additional_chars)
    test_df["cleaned_text"] = test_df["cleaned_text"].apply(remove_additional_chars)



def findall(text, entity):
    entity_length = len(entity)
    result = []
    begin = 0
    if not entity:
        return result
    text = text.lower()
    entity = entity.lower()
    while True:
        pos = text.find(entity, begin)
        if pos != -1:
            result.append((pos, pos+entity_length))
            begin = pos + entity_length
        else:
            return result


def create_tags(tokens, entities):
    tags = ['O'] * len(tokens)
    has_entity = False
    for entity in entities:
        # print('entity:', entity)
        for begin, end in findall(tokens, entity):
            tags[begin] = 'B'
            has_entity = True
            for i in range(begin+1, end):
                tags[i] = 'I'
    # print(tags)
    return tags, has_entity


def create_data(data, output_filename, important_chars, is_evaluate, keep_none):
    line = 0
    with open(output_filename, 'w', encoding='utf-8') as f:
        for idx, text, title, entities in zip(data['id'], data['cleaned_text'], data['cleaned_title'], data['unknownEntities']):
            # print('---------------line:', line)
            entities = entities.split(';')
            sub_texts = []
            offsets = [0]
            title = set(ch for ch in title)

            segment = text
            while len(segment) > MAX_SEQ_LEN:
                sub_texts.append(segment[:MAX_SEQ_LEN])
                comma_pos = segment.find('，', MAX_SEQ_LEN*4//5, MAX_SEQ_LEN)
                if comma_pos == -1:
                    comma_pos = MAX_SEQ_LEN*4//5
                segment = segment[comma_pos:]
                offsets.append(comma_pos)
            else:
                sub_texts.append(segment)
            # print(sub_texts)

            for sub_text, offset in zip(sub_texts, offsets):
                sub_text = sub_text.strip()
                if not sub_text:
                    continue
                if not is_evaluate and len(sub_text) < 6:
                    continue

                tags, has_entity = create_tags(sub_text, entities)
                if not is_evaluate and not keep_none and not has_entity:
                    continue
                f.write('^'*10)
                f.write(idx)
                f.write('\n')
                print(sub_text)
                for i, (char, tag) in enumerate(zip(sub_text, tags)):
                    in_title = 1 if char in title else 0
                    important = 1 if char in important_chars else 0
                    is_lower = 1 if str.islower(char) else 0
                    is_upper = 1 if str.isupper(char) else 0
                    is_num = 1 if str.isnumeric(char) else 0
                    is_sign = 1 if char in extra_chars else 0
                    rel_pos = (i + offset) / len(text)
                    abs_pos = 1 if (i + offset < 100 or i + offset > len(text) - 100) else 0
                    f.write(f'{char} {tag} {in_title} {important} {is_lower} {is_upper} {is_num} {is_sign} {rel_pos} {abs_pos}\n')
                f.write('$'*10)
                f.write('\n')
            line += 1


def collect_important_chars(entities_column):
    char_counts = Counter()
    for entities in entities_column:
        entities = entities.replace(';', '')
        if not entities:
            continue
        char_counts += Counter(entities)
    # print(char_counts)

    return set(v[0] for v in char_counts.items() if v[1] > 10 and not str.isascii(v[0]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir')
    parser.add_argument('--max_seq_len', default=510, type=int)
    parser.add_argument('--keep_none', default=False, action='store_true')

    args = parser.parse_args()
    MAX_SEQ_LEN = args.max_seq_len

    train_data = pd.read_csv('./round2_data/Train_Data.csv', sep=',', dtype=str, encoding='utf-8')
    test_data = pd.read_csv('./round2_data/Test_Data.csv', sep=',', dtype=str, encoding='utf-8')

    train_data.fillna('', inplace=True)
    test_data.fillna('', inplace=True)

    train_data['cleaned_text'] = train_data['text'].apply(clean)
    train_data['cleaned_title'] = train_data['title'].apply(clean)
    test_data['cleaned_text'] = test_data['text'].apply(clean)
    test_data['cleaned_title'] = test_data['title'].apply(clean)
    test_data['unknownEntities'] = ''

    important_chars = collect_important_chars(train_data['unknownEntities'])
    # print(important_chars)
    remove_chars(train_data, test_data)

    train_data = train_data.sample(frac=1, random_state=2019).reset_index(drop=True)
    dev_data = train_data.tail(100)
    train_data = train_data.head(train_data.shape[0]-100)

    create_data(train_data, '{}/train.txt'.format(args.output_dir), important_chars, False, keep_none=args.keep_none)

    create_data(dev_data, '{}/dev.txt'.format(args.output_dir), important_chars, True, keep_none=args.keep_none)

    create_data(test_data, '{}/test.txt'.format(args.output_dir), important_chars, True, keep_none=args.keep_none)