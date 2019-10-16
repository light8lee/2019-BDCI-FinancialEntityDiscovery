import torch as t
import pickle
import numpy as np
from torch.utils.data import Dataset
import scipy.sparse as sp
from proj_utils.graph import sparse_scipy2torch, get_laplacian
from proj_utils import POS_FLAGS
import sys


class GraphDataset(Dataset):
    def __init__(self, fea_file, postions):
        self.fea_file = fea_file
        self.postions = postions

    def __len__(self):
        return len(self.postions)
        # return 100

    def __getitem__(self, idx):
        pos = self.postions[idx]
        self.fea_file.seek(pos)
        feature = pickle.load(self.fea_file)
        return feature


def collect_single(batch):
    # raise ValueError("{}".format(batch))
    idx, batch_input_ids, batch_masks, batch_tag_ids, batch_inputs, batch_flag_ids, batch_bound_ids, batch_extra = zip(*batch)

    batch_input_ids = t.from_numpy(np.array(batch_input_ids)).long()
    batch_masks = t.from_numpy(np.array(batch_masks)).float()
    batch_tag_ids = t.from_numpy(np.array(batch_tag_ids)).long()
    # batch_flag_ids = t.from_numpy(np.array(batch_flag_ids)).long()  # [b, t]
    batch_oh_flags = []
    for flag_ids in batch_flag_ids:
        flag_ids = t.from_numpy(np.array(flag_ids)).long().unsqueeze(-1)  # [t, 1]
        # raise ValueError(f'{flag_ids.shape}')
        batch_oh_flags.append(t.zeros(flag_ids.shape[0], len(POS_FLAGS)).scatter_(1, flag_ids, 1))
    batch_oh_flags = t.stack(batch_oh_flags, 0)

    batch_oh_bounds = []
    for bound_ids in batch_bound_ids:
        bound_ids = t.from_numpy(np.array(bound_ids)).long().unsqueeze(-1)  # [t, 1]
        # raise ValueError(f'{flag_ids.shape}')
        batch_oh_bounds.append(t.zeros(bound_ids.shape[0], 6).scatter_(1, bound_ids, 1))
    batch_oh_bounds = t.stack(batch_oh_bounds, 0)

    try:
        batch_extra = t.from_numpy(np.array(batch_extra)).float()
    except:
        print(batch_extra, file=sys.stderr)
        exit(0)

    return  idx, batch_input_ids, batch_masks, batch_tag_ids, batch_inputs, batch_oh_flags, batch_oh_bounds, batch_extra

class SquadDataset(Dataset):
    def __init(self, all_input_ids, all_input_mask, all_segment_ids,
               all_example_index, all_cls_index, all_p_mask, all_input_idxs):
        self.all_input_ids = all_input_ids
        self.all_input_mask = all_input_mask
        self.all_segment_ids = all_segment_ids
        self.all_example_index = all_example_index
        self.all_cls_index = all_cls_index
        self.all_p_mask = all_p_mask
        self.all_input_idxs = all_input_idxs

    def __len__(self):
        return self.all_input_ids.shape[0]

    def __getitem__(self, idx):
        return (self.all_input_ids[idx], self.all_input_mask[idx], self.all_segment_ids[idx],
                self.all_example_index[idx], self.all_cls_index[idx], self.all_p_mask[idx], self.all_input_idxs[idx])


def collect_squad(batch):
    # raise ValueError("{}".format(batch))
    (input_ids, input_masks, segment_ids,
     example_indexs, cls_indexs, p_masks, input_idxs) = zip(*batch)

    input_ids = torch.stack(input_ids, 0)
    input_masks = torch.stack(input_masks, 0)
    segment_ids = torch.stack(segment_ids, 0)
    example_indexs = torch.stack(example_indexs, 0)
    cls_indexs = torch.stack(cls_indexs, 0)
    p_masks = torch.stack(p_masks, 0)

    return (input_ids, input_masks, segment_ids, example_indexs, cls_indexs, p_masks, input_idxs)