import os
import pandas as pd
import argparse
from collections import Counter


REMOVE = {
    '3点钟XMX', '51天', 'ABC123Group', 'APP有鱼股票',
    'ATM取款机', 'BENZ', 'EXPASSETLtd', 'IXX.com',
    'Miss摩', 'NOSPPT', 'NOSProjectPPTNOS', 'NOSZ',
    'One', 'PLUS', 'Python', 'TPU',
    'Token', 'TrustWalletdApps', 'ZG.comCEO', 'bangquan1',
    'coinsupercoinsuper', 'fengxian', 'hyb华赢宝', 'pandorasdefi',
    'tvb', 'wai汇', 'weixin', '中国环球币',
    '云联惠心未来', '仟易商城微交易微交易平台微交易', '以太猴模式',
    '信广', '刷脸支付', '加密投资银行', '加密货币投资银行',
    '卓逸集', '博安杰', '哪划算苏州哪划算网络公司', '外汇110网',
    '外汇期货', '大众创业万众创新', '大众创业商机', '天乐购',
    '小闪贷鄞州银行小闪贷', '山东再担', '张誉发', '微众银行',
    '微信贷款', '我投资', '晋坤农畜交易50ETFbritrading元银汇宝', '晋江农商银行浙江金华金融服务上门行', '本体',
    '泉州华美整形医院', '炒wai汇', '牛汇BFS牛汇', 
    '祥泰彩印包装', '积分兑换', '立诚贷', '维quan', '贝尔您',
    '贵金属伦敦金伦敦银', '超级富豪', '钻石', '铜柚子', '银柚子',
    '黄金0', '黄金wai汇'
}


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