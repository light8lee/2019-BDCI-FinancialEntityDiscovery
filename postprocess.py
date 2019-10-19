import pandas as pd
import argparse
import os
import pickle
import re
from textrank4zh import TextRank4Keyword
import create_data

BAD_CASES = {
    'http',
    '中国',
    '日本',
    '韩国',
    '美国',
    'app',
    '京东金融',
    '5g',
    '5G'
}

ONLY_NUM = re.compile(r'^\d+$')
INVALID = re.compile(r'[,▌丨\u200b!#$%&*+./:;<=>?@\[\\\]^_`{|}~！#￥？《》{}“”，：‘’。·、；【】]')
train_data = pd.read_csv('./data/Train_Data.csv', sep=',', dtype=str, encoding='utf-8')
train_data.fillna('', inplace=True)
train_entities = set()
for entities in train_data['unknownEntities']:
    entities = entities.split(';')
    train_entities.update(entities)
train_entities.remove('')
TR4K = TextRank4Keyword()
ENGLISH = re.compile(r'^[a-zA-Z0-9.+-]+$')

def clean_samples(samples):
    samples['cleaned_text'] = samples['text'].apply(create_data.clean)
    samples['cleaned_title'] = samples['title'].apply(create_data.clean)


def filter(entities, invalid_entities=None):
    entities = entities.split(';')
    new_entites = []
    for entity in entities:
        if len(entity) < 2:
            continue
        if '（' in entity and '）' not in entity:
            continue
        if '（' not in entity and '）' in entity:
            continue
        if '(' in entity and ')' not in entity:
            continue
        if '(' not in entity and ')' in entity:
            continue
        if ONLY_NUM.match(entity):
            continue
        if INVALID.search(entity):
            continue
        if invalid_entities:
            if entity in invalid_entities:
                continue
            # skip = False
            # for invalid_entity in invalid_entities:
            #     if entity in invalid_entity or invalid_entity in entity:  # FIXME
            #         skip = True
            #         break
            # if skip:
            #     continue
        if entity in train_entities or entity in BAD_CASES:
            continue
        new_entites.append(entity)
    # entities = new_entites
    # new_entites = []
    # for i in range(len(entities)):
    #     current = entities[i]
    #     valid = True
    #     if ENGLISH.match(current):
    #         for j in range(len(entities)):
    #             if i == j:
    #                 continue
    #             other = entities[j]
    #             if ENGLISH.match(other) and (other.find(current)!=-1) and (len(other)-len(current)<=3):
    #                 if other.startswith(current):
    #                     additions = set(other) - set(current)
    #                     for v in additions:
    #                         if v.isupper():
    #                             valid = True
    #                             break
    #                 elif other.endswith(current):
    #                     valid = False
    #                 else:
    #                     valid = False
    #             if not valid:
    #                 break
    #     if valid:
    #         new_entites.append(current)
    return ';'.join(new_entites)


def extract_keywords(phase):
    TR4K.analyze(phase, window=4)
    keywords = [item.word for item in TR4K.get_keywords(20, word_min_len=2)]
    entities = set()
    used = set()
    for i in range(len(keywords)):
        for j in range(i+1, len(keywords)):
            candidate = keywords[i] + keywords[j]
            if candidate in phase:
                entities.add(candidate)
                used.add(keywords[i])
                used.add(keywords[j])
            candidate = keywords[j] + keywords[i]
            if candidate in phase:
                entities.add(candidate)
                used.add(keywords[i])
                used.add(keywords[j])
    entities.update(keywords[:3])
    entities = entities - used
    return ';'.join(entities)


def convert_to_submit(name, invalid_entities=None):
    input_filename = os.path.join('outputs', name, 'submit.csv')
    preds = pd.read_csv(input_filename, sep=',', index_col='id')
    sample = pd.read_csv('data/Test_Data.csv', sep=',', index_col='id')
    assert sample.shape[0] == preds.shape[0]
    assert sample.shape[0] == len(sample.index & preds.index)
    print('not in sample', set(preds.index)-set(sample.index))
    print('not in preds', set(sample.index)-set(preds.index))
    outputs = preds.reindex(sample.index)
    outputs.fillna('', inplace=True)
    sample.fillna('', inplace=True)
    outputs['unknownEntities'] = outputs['unknownEntities'].apply(lambda v: filter(v, invalid_entities))
    # clean_samples(sample)
    # for idx, row in outputs.iterrows():
    #     if not row['unknownEntities']:
    #         row['unknownEntities'] = extract_keywords(sample.at[idx, 'cleaned_title']+';'+sample.at[idx, 'cleaned_text'][:50])
    output_filename = os.path.join('submits', '{}.csv'.format(name))
    outputs.to_csv(output_filename)


def merge_and_convert_to_submit(crf_name, squad_name, invalid_entities=None):
    crf_filename = os.path.join('outputs', crf_name, 'submit.csv')
    crf_preds = pd.read_csv(crf_filename, sep=',', index_col='id')
    squad_filename = os.path.join('outputs', squad_name, 'submit.csv')
    squad_preds = pd.read_csv(squad_filename, sep=',', index_col='id')
    sample = pd.read_csv('data/Test_Data.csv', sep=',', index_col='id')
    print('crf:', crf_preds.shape)
    print('squad:', squad_preds.shape)
    assert sample.shape[0] == crf_preds.shape[0] == squad_preds.shape[0]
    assert sample.shape[0] == len(sample.index & crf_preds.index) == len(sample.index & squad_preds.index)
    crf_preds.fillna('', inplace=True)
    squad_preds.fillna('', inplace=True)
    sample.fillna('', inplace=True)
    crf_outputs = crf_preds.reindex(sample.index)
    squad_outputs = squad_preds.reindex(sample.index)
    for (index, crf_row) in crf_outputs.iterrows():
        print(crf_row)
        # if not crf_row['unknownEntities']:
        #     crf_row['unknownEntities'] = squad_outputs.at[index, 'unknownEntities']
        entities = set()
        entities.add('')
        entities.update(crf_row['unknownEntities'].split(';'))
        entities.update(squad_outputs.at[index, 'unknownEntities'].split(';'))
        entities.remove('')
        crf_row['unknownEntities'] = ';'.join(entities)
        print(crf_row)

    crf_outputs['unknownEntities'] = crf_outputs['unknownEntities'].apply(lambda v: filter(v, invalid_entities))
    # clean_samples(sample)
    # for idx, row in crf_outputs.iterrows():
    #     if not row['unknownEntities']:
    #         row['unknownEntities'] = extract_keywords(sample.at[idx, 'cleaned_title']+';'+sample.at[idx, 'cleaned_text'][:50])
    output_filename = os.path.join('submits', '{}-{}.csv'.format(crf_name, squad_name))
    crf_outputs.to_csv(output_filename)


def merge_and_convert_to_submit_v2(output_name, crf_names, squad_name, invalid_entities=None):
    crf_filenames = [os.path.join('outputs', crf_name, 'submit.csv') for crf_name in crf_names]
    crf_preds = [pd.read_csv(crf_filename, sep=',', index_col='id') for crf_filename in crf_filenames]
    squad_filename = os.path.join('outputs', squad_name, 'submit.csv')
    squad_pred = pd.read_csv(squad_filename, sep=',', index_col='id')
    sample = pd.read_csv('data/Test_Data.csv', sep=',', index_col='id')
    # print('crf:', crf_preds.shape)
    # print('squad:', squad_preds.shape)
    for crf_pred in crf_preds:
        assert sample.shape[0] == crf_pred.shape[0] == squad_pred.shape[0]
        assert sample.shape[0] == len(sample.index & crf_pred.index) == len(sample.index & squad_pred.index)
        crf_pred.fillna('', inplace=True)
    squad_pred.fillna('', inplace=True)
    sample.fillna('', inplace=True)
    crf_outputs = [crf_pred.reindex(sample.index) for crf_pred in crf_preds]
    squad_output = squad_pred.reindex(sample.index)
    for (index, crf_row) in squad_output.iterrows():
        # print(crf_row)
        # if not crf_row['unknownEntities']:
        #     crf_row['unknownEntities'] = squad_outputs.at[index, 'unknownEntities']
        entities = set()
        entities.add('')
        entities.update(crf_row['unknownEntities'].split(';'))
        # entities.update(squad_outputs.at[index, 'unknownEntities'].split(';'))
        crf_entities = set()
        for crf_output in crf_outputs:
            tmp_entites = set(crf_output.at[index, 'unknownEntities'].split(';'))
            if not tmp_entites:
                continue
            if not crf_entities:
                crf_entities = tmp_entites
            else:
                crf_entities = crf_entities & tmp_entites
            # entities.update(tmp_entites)
        entities.update(crf_entities)
        entities.remove('')
        crf_row['unknownEntities'] = ';'.join(entities)
        print(crf_row)

    squad_output['unknownEntities'] = squad_output['unknownEntities'].apply(lambda v: filter(v, invalid_entities))
    # clean_samples(sample)
    # for idx, row in crf_outputs.iterrows():
    #     if not row['unknownEntities']:
    #         row['unknownEntities'] = extract_keywords(sample.at[idx, 'cleaned_title']+';'+sample.at[idx, 'cleaned_text'][:50])
    squad_output.to_csv(output_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--crf_model', type=str, default='')
    parser.add_argument('--squad_model', type=str, default='')
    parser.add_argument('--kfold', type=int, default=0)
    # parser.add_argument('--invalid_entities', type=str, default='')
    # parser.add_argument('--fold', type=int, default=-1)
    args = parser.parse_args()
    # if args.invalid_entities:
    #     if args.invalid_entities.find('invalid_entities') != -1:  # 指定了文件
    #         with open(args.invalid_entities, 'rb') as f:
    #             invalid_entities = pickle.load(f)
    #             invalid_entities = set(v for v in invalid_entities if len(v) > 1)
    #     else:  # 指定了目录
    #         if args.fold > 0:  # k折的结果
    #             invalid_entities = set()
    #             for i in range(args.fold):
    #                 with open(os.path.join(args.invalid_entities, f'fold{i}', 'invalid_entities'), 'rb') as f:
    #                     invalid_entities.update(pickle.load(f))
    #             invalid_entities = set(v for v in invalid_entities if len(v) > 1)
    #         else:  # 单个结果
    #             with open(os.path.join(args.invalid_entities, 'invalid_entities'), 'rb') as f:
    #                 invalid_entities = pickle.load(f)
    #                 invalid_entities = set(v for v in invalid_entities if len(v) > 1)
    # else:
    #     invalid_entities = None
    if not args.crf_model and not args.squad_model:
        raise ValueError("Should be at least one model")
    elif args.crf_model and not args.squad_model:
        convert_to_submit(args.crf_model,)
    elif args.squad_model and not args.crf_model:
        convert_to_submit(args.squad_model,)
    else:
        output_filename = os.path.join('submits', f'{args.crf_model}-{args.squad_model}.csv')
        crf_names = args.crf_model
        crf_names = crf_names.split(',')
        if args.kfold:
            crf_names = [os.path.join(crf_names[0], f'fold{i}') for i in range(args.kfold)]
        merge_and_convert_to_submit_v2(output_filename, crf_names, args.squad_model,)
