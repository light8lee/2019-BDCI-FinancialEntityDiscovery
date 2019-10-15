import numpy as np

def get_unit_idx(unit_idxs, unit):
    if unit not in unit_idxs:
        unit = '[UNK]'
    return unit_idxs[unit]

class Unit2idx:
    def __init__(self, unit_idxs):
        self.unit_idxs = unit_idxs

    def __call__(self, unit):
        return get_unit_idx(self.unit_idxs, unit)


BIO_ID2TAG = ['O', 'B', 'I']
BIO_TAG2ID = {name: idx for idx, name in enumerate(BIO_ID2TAG)}

POS_FLAGS = ['[PAD]', '[CLS]', '[SEP]', 
             'ag', 'a', 'ad', 'an', 'b', 'c', 'dg',
             'd', 'e', 'eng', 'f', 'g', 'h', 'i', 'j', 'k',
             'l', 'm', 'ng', 'n', 'nr', 'ns', 'nt', 'nz', 
             'o', 'p', 'q', 'r', 's', 'tg', 't', 'u', 'un', 
             'vg', 'v', 'vd', 'vn', 'w', 'x', 'y', 'z']
POS_FLAGS_TO_IDS = {flag: i for i, flag in enumerate(POS_FLAGS)}

WORD_BOUNDS = ['[PAD]', '[CLS]', '[SEP]', 'B', 'I', 'S']
WORD_BOUNDS_TO_IDS = {bound: i for i, bound in enumerate(WORD_BOUNDS)}