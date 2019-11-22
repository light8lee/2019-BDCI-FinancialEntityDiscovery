import os
import pandas as pd
import argparse
from collections import Counter
with open('data/bad_cases.txt', encoding='utf-8') as f:
    REMOVE = set(line.strip() for line in f)

def vote_v1():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('models', type=str, default='')
    # args = parser.parse_args()
    models = 'roberta_ext_v25,roberta_ext_m11'
    test = pd.read_csv('round2_data/Test_Data.csv', sep=',')
    test.fillna('', inplace=True)
    
    submits = [pd.read_csv(os.path.join('submits', f"{name}.csv")) for name in models.split(',')]
    for submit in submits:
        submit.fillna('', inplace=True)
    num_submit = len(submits)
    size = submits[0].shape[0]
    output = pd.DataFrame()
    output['id'] = submits[0]['id']
    output['unknownEntities'] = ''

    for i in range(size):
        entity_counter = Counter()
        for submit in submits:
            tmp_entities = submit.at[i, 'unknownEntities']
            if not tmp_entities:
                continue
            tmp_counter = Counter(tmp_entities.split(';'))
            entity_counter.update(tmp_counter)
        entity_set = set()
        new_entity_counter = Counter()
        for entity in entity_counter:
            length = len(entity)
            if entity.startswith('XX'):
                continue
            if length>=6 and length%2==0:
                same = True
                for pos in range(length//2):
                    if entity[pos] != entity[pos+length//2]:
                        same = False
                        break
                if same:
                    print(entity)
                    new_entity_counter[entity[:length//2]] += entity_counter[entity]
                else:
                    new_entity_counter[entity] += entity_counter[entity]
            else:
                new_entity_counter[entity] += entity_counter[entity]
        for entity, count in new_entity_counter.items():
            if entity in REMOVE:
                continue
            if entity not in test.at[i, 'text'] and entity not in test.at[i, 'title']:
                print(f'line: {i}, not in: {entity}')
                continue
            if count >= 1:
                entity_set.add(entity)
        output.at[i, 'unknownEntities'] = ';'.join(entity_set)
        # print(output.head(4))

    output.to_csv(os.path.join('submits', f"11-21.csv"), index=False)


vote_v1()