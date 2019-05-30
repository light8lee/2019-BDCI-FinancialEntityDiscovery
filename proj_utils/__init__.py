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
