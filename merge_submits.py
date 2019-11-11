import os
import pandas as pd
import argparse

def merge_v1():
    parser = argparse.ArgumentParser()
    parser.add_argument('models', type=str, default='')
    args = parser.parse_args()

    submits = [pd.read_csv(os.path.join('submits', f"{name}.csv")) for name in args.models.split(',')]
    for submit in submits:
        submit.fillna('', inplace=True)
    size = submits[0].shape[0]
    output = pd.DataFrame()
    output['id'] = submits[0]['id']
    output['unknownEntities'] = ''

    for i in range(size):
        entity_set = set()
        entity_set.add('')
        for submit in submits:
            entity_set.update(submit.at[i, 'unknownEntities'].split(';'))
        print(entity_set)
        entity_set.remove('')
        output.at[i, 'unknownEntities'] = ';'.join(entity_set)

    output.to_csv(os.path.join('submits', f"{args.models}.csv"), index=False)


def merge_v2():
    parser = argparse.ArgumentParser()
    parser.add_argument('base', type=str, default='')
    parser.add_argument('add', type=str, default='')
    args = parser.parse_args()

    base_submit = pd.read_csv(os.path.join('submits', f"{args.base}.csv"))
    add_submit = pd.read_csv(os.path.join('submits', f"{args.add}.csv"))
    base_submit.fillna('', inplace=True)
    add_submit.fillna('', inplace=True)

    for i in range(base_submit.shape[0]):
        if not base_submit.at[i, 'unknownEntities']:
            base_submit.at[i, 'unknownEntities'] = add_submit.at[i, 'unknownEntities']

    base_submit.to_csv(os.path.join('submits', f"{args.base}-{args.add}.csv"), index=False)

merge_v1()
# merge_v2()