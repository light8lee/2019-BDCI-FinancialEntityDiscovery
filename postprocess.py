import pandas as pd
import jieba
import jieba.posseg as pseg
import math
import argparse
import os
import pickle
import re
from textrank4zh import TextRank4Keyword
import create_data

DOMAINS = set()
with open('data/domains.txt') as f:
    for line in f:
        line = line.strip()
        if line.endswith('市'):
            DOMAINS.add((line, line[:-1]))
        elif line.endswith('省'):
            DOMAINS.add((line, line[:-1]))
        else:
            DOMAINS.add((line, line))

TOGGLES = {
    '平台', '国际',
}

COMPLETES = {
    '限公司', '司',
}

PARTS = {
    '交易中心',
    '有限公司',
}

BAD_CASES = {
    'http', 'HTTP', '中国', '日本',
    '韩国', '美国', '京东金融', '5g', '5G',
    'IMG', 'admin', 'QQ', 'VIP', '保险', '天乐购'
    'IP', 'font-family', 'font', 'family', 'font-size',
    'wifi', '24x7', '小视频', '51天', '上证个股期权', 'ind',
    'VP', '财富', '云交易', '流浪汉', '实干家', '空想家', '梦想家',
    '高汇高汇', '稀万acy', 'pnk跑', 'freedigitalbankingservicesGKBank'
}

BAD_HEADS = {
    'iphone',
    'iPhone'
}

BAD_TAILS = {
    'app', 'App', 'APP', '-',
    '美元', '日元', '人民币', '.'
}

REPLACE = re.compile(r'[*“,/#?]')
ONLY_NUM = re.compile(r'^\d+$')
INVALID = re.compile(r'[\s▌丨\u200b!$%:;<=>@\[\\\]^_`{|}~！#￥？《》{}”，：‘’。、；【】]')
train_data = pd.read_csv('./round2_data/Train_Data.csv', sep=',', dtype=str, encoding='utf-8')
train_data.fillna('', inplace=True)
train_entities = set()
for entities in train_data['unknownEntities']:
    entities = entities.split(';')
    train_entities.update(entities)
train_entities.remove('')
TR4K = TextRank4Keyword()
ENGLISH = re.compile(r'^[a-zA-Z0-9]+$')

def clean_samples(samples):
    samples['cleaned_text'] = samples['text'].apply(create_data.clean)
    samples['cleaned_title'] = samples['title'].apply(create_data.clean)


def clean_rc_result(entities):
    entities = entities.split(';')
    new_entities = []
    for entity in entities:
        is_bad = False
        if ENGLISH.match(entity):
            is_bad = True
        if is_bad:
            continue
        new_entities.append(entity)
    return ';'.join(new_entities)


def filter(entities, invalid_entities=None):
    entities = entities.split(';')
    new_entites = []
    for entity in entities:
        is_bad = False
        if len(entity) < 2:
            continue
        if '（' in entity and '）' not in entity:
            print(f"括号不完全：{entity}")
            continue
        if '（' not in entity and '）' in entity:
            print(f"括号不完全：{entity}")
            continue
        if '(' in entity and ')' not in entity:
            print(f"括号不完全：{entity}")
            continue
        if '(' not in entity and ')' in entity:
            print(f"括号不完全：{entity}")
            continue
        if ONLY_NUM.match(entity):
            print(f"纯数字：{entity}")
            continue
        if INVALID.search(entity):
            print(f"包含不正常符号：{entity}")
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
        new_entity = REPLACE.sub('', entity)
        if entity != new_entity:
            print(f"去除特殊字符：{entity} => {new_entity}")
            entity = new_entity

        if entity in train_entities or entity in BAD_CASES:
            print(f"已在训练集中或错例：{entity}")
            continue
        # for oentity in train_entities:
        #     loentity = oentity.lower()
        #     if loentity in entity.lower() and not str.islower(loentity):
        #         print(f"已包含训练集中或错例：{entity} : {oentity}")
        #         is_bad = True
        #         break
        # if is_bad:
        #     continue

        for bad_head in BAD_HEADS:
            if entity.startswith(bad_head):
                print(f"错误开头：{entity}")
                is_bad = True
                break
        if is_bad:
            continue
        for bad_tail in BAD_TAILS:
            if entity.endswith(bad_tail):
                print(f"错误结尾：{entity}")
                is_bad = True
                break
        if is_bad:
            continue
        for full_domain, mini_domain in DOMAINS:
            if entity == full_domain or entity == mini_domain:
                print(f"地名：{entity}")
                is_bad = True
                break
        if is_bad:
            continue
        if ENGLISH.match(entity) and len(entity) >= 30:
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


def keep_topk(outputs, sample, k=5):
    for i in range(outputs.shape[0]):
        entities = outputs.at[i, 'unknownEntities'].split(';')
        if not entities:
            continue
        scores = []
        origin_text = sample.at[i, 'text']
        for entity in entities:
            if not entity:  # 可能会有问题，有些实体在清洗后才出现
                continue
            count = origin_text.count(entity)
            pos = origin_text.find(entity)
            size = len(entity)
            # print(count, pos, size)
            score = 10 * count + math.pow(10, 1/(2+pos)) + math.log10(size)  # count+1/(2+pos)+math.log10(size)
            scores.append(
                (entity, score)
            )
            scores.sort(key=lambda v: v[1], reverse=True)
            scores = scores[:k]
            outputs.at[i, 'unknownEntities'] = ';'.join(v[0] for v in scores)


def do_rule(outputs, sample):
    for i in range(outputs.shape[0]):
        entities = outputs.at[i, 'unknownEntities'].split(';')
        if not entities:
            continue
        origin_text = sample.at[i, 'text']
        new_entities = set(entities)
        for entity in entities:
            if not entity:
                continue
            if INVALID.search(entity):
                continue
            need_remove = False
            origin_entity = entity

            # 去除地名
            # for full_domain, mini_domain in DOMAINS:
            #     if entity.startswith(full_domain):
            #         new_entity = entity[len(full_domain):]
            #         print(f'删除地名：{entity} => {new_entity}')
            #         entity = new_entity
            #     if entity.startswith(mini_domain):
            #         new_entity = entity[len(mini_domain):]
            #         print(f'删除地名：{entity} => {new_entity}')
            #         entity = new_entity
            
            # 去除平台
            # if entity.endswith('平台'):
            #     new_entity = entity[:-2]
            #     print(f'删除平台：{entity} => {new_entity}')
            #     entity = new_entity
            
            # 补全 有限公司
            if entity.endswith('有限公'):
                new_entity = entity + '司'
                if new_entity in origin_text:
                    print(f'补全有限公司：{entity} => {new_entity}')
                    new_entities.add(new_entity)
                    need_remove = True
            if entity.endswith('有限'):
                new_entity = entity + '公司'
                if new_entity in origin_text:
                    print(f'补全有限公司：{entity} => {new_entity}')
                    new_entities.add(new_entity)
                    need_remove = True
            if entity.endswith('有'):
                new_entity = entity + '限公司'
                if new_entity in origin_text:
                    print(f'补全有限公司：{entity} => {new_entity}')
                    new_entities.add(new_entity)
                    need_remove = True
            
            # 去除 有限公司、公司、集团
            # if entity.endswith('有限公司'):
            #     new_entity = ''
            #     for word, flag in pseg.lcut(entity):
            #         if len(new_entity) + len(word) <= 5:
            #             new_entity = new_entity + word
            #         else:
            #             break
            #     if new_entity:
            #         print(f'去除有限公司：{entity} => {new_entity}')
            #         entity = new_entity
            # if entity.endswith('公司'):
            #     new_entity = ''
            #     for word, flag in pseg.lcut(entity[:-2]):
            #         if len(new_entity) + len(word) <= 5:
            #             new_entity = new_entity + word
            #         else:
            #             break
            #     print(f'去除公司：{entity} => {new_entity}')
            #     entity = new_entity
            # if entity.endswith('集团'):
            #     new_entity = entity[:-2]
            #     print(f'去除集团：{entity} => {new_entity}')
            #     entity = new_entity

            # 提取括号前的东西
            if '（' in entity and '）' not in entity:
                new_entity = entity[:entity.find('（')]
                print(f'删除括号：{entity} => {new_entity}')
                new_entities.add(new_entity)
                need_remove = True
            if '(' in entity and ')' not in entity:
                new_entity = entity[:entity.find('(')]
                print(f'删除括号：{entity} => {new_entity}')
                new_entities.add(new_entity)
                need_remove = True

            if need_remove:
                new_entities.remove(origin_entity)
                    
            # 尝试补全括号
            if '（' in entity and '）' not in entity:
                pos = origin_text.find(entity)
                if pos != -1:
                    start = pos + len(entity)
                    end = start + 5
                    pos = origin_text[start:end].find('）')
                    if pos != -1:
                        new_entity = entity + origin_text[start:start+pos+1]
                        new_entities.add(new_entity)
                        print(f'括号：', origin_text[start:end])
                        print(f'补全括号：{entity} => {new_entity}')
                        need_remove = True
                        new_entity = new_entity[new_entity.find('（')+1:new_entity.find('）')]
                        print(f'=> {new_entity}')
                        new_entities.add(new_entity)
            if '(' in entity and ')' not in entity:
                pos = origin_text.find(entity)
                if pos != -1:
                    start = pos + len(entity)
                    end = start + 5
                    pos = origin_text[start:end].find(')')
                    if pos != -1:
                        new_entity = entity + origin_text[start:start+pos+1]
                        new_entities.add(new_entity)
                        print(f'括号：', origin_text[start:end])
                        print(f'补全括号：{entity} => {new_entity}')
                        need_remove = True
                        new_entity = new_entity[new_entity.find('(')+1:new_entity.find(')')]
                        print(f'=> {new_entity}')
                        new_entities.add(new_entity)
            
            # 补全 有限公司
            # for complete in COMPLETES:
            #     new_entity = entity + complete
            #     if new_entity in origin_text:
            #         new_entities.remove(entity)
            #         new_entities.add(new_entity)
            #         print(f'补全有限公司：{entity} => {new_entity}')
            #         entity = new_entity
            # 
            # # 补全 后缀
            # for part in PARTS:
            #     pos = origin_text.find(entity)
            #     start = pos + len(entity)
            #     end = start + 10
            #     pos = origin_text[start:end].find(part)
            #     if pos != -1:
            #         new_entity = entity + origin_text[start:start+pos+len(part)]
            #         new_entities.add(new_entity)
            #         print(f'后缀：', origin_text[start:end])
            #         print(f'补全后缀：{entity} => {new_entity}')
            #         entity = new_entity

            # 尝试增加或删除结尾的一些相关词
            # for toggle in TOGGLES:
            #     if entity.endswith(toggle):
            #         # new_entity = entity[:-len(toggle)]
            #         # if len(new_entity) > 3:
            #         #     new_entities.add(new_entity)
            #         #     print(f'删除结尾：{entity} => {new_entity}')
            #         pass
            #     else:
            #         new_entity = entity + toggle
            #         if new_entity in origin_text:
            #             new_entities.add(new_entity)
            #             print(f'增加结尾：{entity} => {new_entity}')
                    
            # # 尝试补全后面的英文
            # if str.islower(entity[-1]) or str.isupper(entity[-1]):
            #     start = origin_text.find(entity)
            #     if start != -1:
            #         end = start + len(entity)
            #         while end < len(origin_text) - 1 and (str.islower(origin_text[end]) or str.isupper(origin_text[end])):
            #             end += 1
            #         new_entity = origin_text[start:end]
            #         if new_entity != entity:
            #             new_entities.remove(entity)
            #             new_entities.add(new_entity)
            #             print(f'补全英文：{entity} => {new_entity}')

            # 尝试删除后面的英文
            # if not ENGLISH.match(entity):
            #     new_entity = entity
            #     while str.islower(new_entity[-1]) or str.isupper(new_entity[-1]):
            #         new_entity = new_entity[:-1]
            #     if new_entity and new_entity != entity:
            #         new_entities.remove(entity)
            #         new_entities.add(new_entity)
            #         print(f'删除英文：{entity} => {new_entity}')
            #         entity = new_entity

            # 尝试增加或删除开头的地名
            # for full_domain, mini_domain in DOMAINS:
            #     if entity.startswith(mini_domain) and not entity.startswith(full_domain):
            #         # new_entity = entity[len(mini_domain):]
            #         # new_entities.add(new_entity)
            #         # print(f'删除地名：{entity} => {new_entity}')
            #         pass
            #     else:
            #         new_entity = full_domain + entity
            #         if new_entity in origin_text:
            #             new_entities.add(new_entity)
            #             print(f'添加地名：{entity} => {new_entity}')
            #         new_entity = mini_domain + entity
            #         if new_entity in origin_text:
            #             new_entities.add(new_entity)
            #             print(f'添加地名：{entity} => {new_entity}')
            
            # # 尝试在开头添加英文
            # new_entity = entity
            # pos = origin_text.find(entity)
            # if pos != -1:
            #     while pos > 0 and (str.islower(origin_text[pos-1]) or str.isupper(origin_text[pos-1])):
            #         new_entity = origin_text[pos-1] + new_entity
            #         pos -= 1
            #     if new_entity != entity:
            #         new_entities.add(new_entity)
            #         print(f'添加开头英文：{entity} => {new_entity}')

        if '' in new_entities:
            new_entities.remove('')
        outputs.at[i, 'unknownEntities'] = ';'.join(new_entities)


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


def convert_to_submit(name, invalid_entities=None, topk=0, is_squad=False):
    input_filename = os.path.join('outputs', name, 'submit.csv')
    preds = pd.read_csv(input_filename, sep=',', index_col='id')
    sample = pd.read_csv('round2_data/Test_Data.csv', sep=',', index_col='id')
    assert sample.shape[0] == preds.shape[0], f"{sample.shape[0]}, {preds.shape[0]}"
    assert sample.shape[0] == len(sample.index & preds.index)
    print('not in sample', set(preds.index)-set(sample.index))
    print('not in preds', set(sample.index)-set(preds.index))
    outputs = preds.reindex(sample.index)
    outputs.fillna('', inplace=True)
    sample.fillna('', inplace=True)
    # clean_samples(sample)
    # for idx, row in outputs.iterrows():
    #     if not row['unknownEntities']:
    #         row['unknownEntities'] = extract_keywords(sample.at[idx, 'cleaned_title']+';'+sample.at[idx, 'cleaned_text'][:50])
    # do_rule(outputs, sample)
    if is_squad:
        outputs['unknownEntities'] = outputs['unknownEntities'].apply(lambda v: clean_rc_result(v))
    outputs['unknownEntities'] = outputs['unknownEntities'].apply(lambda v: filter(v, invalid_entities))
    if topk > 0:
        keep_topk(outputs, sample, topk)
    output_filename = os.path.join('submits', '{}.csv'.format(name.replace('/', '-')))
    outputs.to_csv(output_filename)


def merge_and_convert_to_submit(crf_name, squad_name, invalid_entities=None, topk=0):
    crf_filename = os.path.join('outputs', crf_name, 'submit.csv')
    crf_preds = pd.read_csv(crf_filename, sep=',', index_col='id')
    squad_filename = os.path.join('outputs', squad_name, 'submit.csv')
    squad_preds = pd.read_csv(squad_filename, sep=',', index_col='id')
    sample = pd.read_csv('round2_data/Test_Data.csv', sep=',', index_col='id')
    print('crf:', crf_preds.shape)
    print('squad:', squad_preds.shape)
    assert sample.shape[0] == crf_preds.shape[0] == squad_preds.shape[0]
    assert sample.shape[0] == len(sample.index & crf_preds.index) == len(sample.index & squad_preds.index)
    crf_preds.fillna('', inplace=True)
    squad_preds.fillna('', inplace=True)
    sample.fillna('', inplace=True)
    crf_outputs = crf_preds.reindex(sample.index)
    squad_outputs = squad_preds.reindex(sample.index)
    for (index, crf_row) in crf_outputs.iterrows():  # FIXME
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

    # clean_samples(sample)
    # for idx, row in crf_outputs.iterrows():
    #     if not row['unknownEntities']:
    #         row['unknownEntities'] = extract_keywords(sample.at[idx, 'cleaned_title']+';'+sample.at[idx, 'cleaned_text'][:50])
    # do_rule(crf_outputs, sample)
    crf_outputs['unknownEntities'] = crf_outputs['unknownEntities'].apply(lambda v: filter(v, invalid_entities))
    if topk > 0:
        keep_topk(crf_outputs, sample, topk)
    output_filename = os.path.join('submits', '{}-{}.csv'.format(crf_name, squad_name))
    crf_outputs.to_csv(output_filename)


def merge_and_convert_to_submit_v2(output_name, crf_names, squad_name, invalid_entities=None, topk=0):
    crf_filenames = [os.path.join('outputs', crf_name, 'submit.csv') for crf_name in crf_names]
    crf_preds = [pd.read_csv(crf_filename, sep=',', index_col='id') for crf_filename in crf_filenames]
    squad_filename = os.path.join('outputs', squad_name, 'submit.csv')
    squad_pred = pd.read_csv(squad_filename, sep=',', index_col='id')
    sample = pd.read_csv('round2_data/Test_Data.csv', sep=',', index_col='id')
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
    squad_output['unknownEntities'] = squad_output['unknownEntities'].apply(lambda v: clean_rc_result(v))
    for index in range(squad_output.shape[0]):
        # print(crf_row)
        # if not crf_row['unknownEntities']:
        #     crf_row['unknownEntities'] = squad_outputs.at[index, 'unknownEntities']
        entities = set()
        entities.add('')
        entities.update(squad_output.at[index, 'unknownEntities'].split(';'))
        # entities.update(squad_outputs.at[index, 'unknownEntities'].split(';'))
        # crf_entities = set()
        for crf_output in crf_outputs:
            tmp_entites = set(crf_output.at[index, 'unknownEntities'].split(';'))
            # if not tmp_entites:
            #     continue
            # if not crf_entities:
            #     crf_entities = tmp_entites
            # else:
            #     crf_entities = crf_entities & tmp_entites
            entities.update(tmp_entites)
        # entities.update(crf_entities)
        entities.remove('')
        squad_output.at[index, 'unknownEntities'] = ';'.join(entities)

    # do_rule(squad_output, sample)
    squad_output['unknownEntities'] = squad_output['unknownEntities'].apply(lambda v: filter(v, invalid_entities))
    if topk > 0:
        keep_topk(squad_output, sample, topk)
    squad_output.to_csv(output_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--crf_model', type=str, default='')
    parser.add_argument('--squad_model', type=str, default='')
    parser.add_argument('--kfold', type=int, default=0)
    parser.add_argument('--topk', type=int, default=0)
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
        convert_to_submit(args.crf_model, topk=args.topk)
    elif args.squad_model and not args.crf_model:
        convert_to_submit(args.squad_model, topk=args.topk, is_squad=True)
    else:
        output_filename = os.path.join('submits', f'{args.crf_model}-{args.squad_model.replace("/", "-")}.csv')
        crf_names = args.crf_model
        crf_names = crf_names.split(',')
        if args.kfold:
            crf_names = [os.path.join(crf_names[0], f'fold{i}') for i in range(args.kfold)]
        merge_and_convert_to_submit_v2(output_filename, crf_names, args.squad_model, topk=args.topk)
