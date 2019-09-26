import pandas as pd
import argparse
import os
import pickle
import re

invalid = re.compile(r'[,▌\u200b〖〗…？，！!\d▲《》.▼☑☑【、“”＂＼＇：】％＃＠＊＆＾￥$\[\]—]')
train_data = pd.read_csv('./data/Train_Data.csv', sep=',', dtype=str, encoding='utf-8')
train_data.fillna('', inplace=True)
train_entities = set()
for entities in train_data['unknownEntities']:
    entities = entities.split(';')
    train_entities.update(entities)
train_entities.remove('')

def filter(entities, invalid_entities):
    entities = entities.split(';')
    new_entites = []
    for entity in entities:
        if len(entity) < 2:
            continue
        if '（' in entity and '）' not in entity:
            continue
        if '（' not in entity and '）' in entity:
            continue
        if '(' in entity and '(' not in entity:
            continue
        if '(' not in entity and '(' in entity:
            continue
        if invalid.search(entity):
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
        if entity in train_entities:
            continue
        new_entites.append(entity)
    new_entites.sort()
    entities = new_entites
    new_entites = []
    for i in range(len(entities)):
        current = entities[i]
        valid = True
        for j in range(i+1, len(entities)):
            other = entities[j]
            if other.find(current) != -1:
                valid = False
                break
        if valid:
            new_entites.append(current)
    return ';'.join(new_entites)


def convert_to_submit(name, invalid_entities):
    input_filename = os.path.join('outputs', name, 'submit.csv')
    preds = pd.read_csv(input_filename, sep=',', index_col='id')
    sample = pd.read_csv('data/Test_Data.csv', sep=',', index_col='id')
    assert sample.shape[0] == preds.shape[0]
    assert sample.shape[0] == len(sample.index & preds.index)
    print('not in sample', set(preds.index)-set(sample.index))
    print('not in preds', set(sample.index)-set(preds.index))
    outputs = preds.reindex(sample.index)
    outputs.fillna('', inplace=True)
    outputs['unknownEntities'] = outputs['unknownEntities'].apply(lambda v: filter(v, invalid_entities))
    output_filename = os.path.join('submits', '{}.csv'.format(name))
    outputs.to_csv(output_filename)


def merge_and_convert_to_submit(crf_name, squad_name, invalid_entities):
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
    crf_outputs = crf_preds.reindex(sample.index)
    squad_outputs = squad_preds.reindex(sample.index)
    for (index, crf_row) in crf_outputs.iterrows():
        print(crf_row)
        if not crf_row['unknownEntities']:
            crf_row['unknownEntities'] = squad_outputs.at[index, 'unknownEntities']
        print(crf_row)

    crf_outputs['unknownEntities'] = crf_outputs['unknownEntities'].apply(lambda v: filter(v, invalid_entities))
    output_filename = os.path.join('submits', '{}-{}.csv'.format(crf_name, squad_name))
    crf_outputs.to_csv(output_filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--crf_model', type=str, default='')
    parser.add_argument('--squad_model', type=str, default='')
    parser.add_argument('--invalid_entities', type=str, default='')
    parser.add_argument('--fold', type=int, default=-1)
    args = parser.parse_args()
    if args.invalid_entities:
        if args.invalid_entities.find('invalid_entities') != -1:  # 指定了文件
            with open(args.invalid_entities, 'rb') as f:
                invalid_entities = pickle.load(f)
                invalid_entities = set(v for v in invalid_entities if len(v) > 1)
        else:  # 指定了目录
            if args.fold > 0:  # k折的结果
                invalid_entities = set()
                for i in range(args.fold):
                    with open(os.path.join(args.invalid_entities, f'fold{i}', 'invalid_entities'), 'rb') as f:
                        invalid_entities.update(pickle.load(f))
                invalid_entities = set(v for v in invalid_entities if len(v) > 1)
            else:  # 单个结果
                with open(os.path.join(args.invalid_entities, 'invalid_entities'), 'rb') as f:
                    invalid_entities = pickle.load(f)
                    invalid_entities = set(v for v in invalid_entities if len(v) > 1)
    else:
        invalid_entities = None
    if not args.crf_model and not args.squad_model:
        raise ValueError("Should be at least one model")
    elif args.crf_model and not args.squad_model:
        convert_to_submit(args.crf_model, invalid_entities)
    elif args.squad_model and not args.crf_model:
        convert_to_submit(args.squad_model, invalid_entities)
    else:
        merge_and_convert_to_submit(args.crf_model, args.squad_model, invalid_entities)
