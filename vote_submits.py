import os
import pandas as pd
import argparse
from collections import Counter

def vote_v1():
    parser = argparse.ArgumentParser()
    parser.add_argument('models', type=str, default='')
    args = parser.parse_args()
    
    submits = [pd.read_csv(os.path.join('submits', f"{name}.csv")) for name in args.models.split(',')]
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
        for entity, count in entity_counter.items():
            if count > num_submit//2:
                entity_set.add(entity)
        output.at[i, 'unknownEntities'] = ';'.join(entity_set)

    output.to_csv(os.path.join('submits', f"{args.models.replace(',', '@')}.csv"), index=False)