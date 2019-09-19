import torch as t
import pickle
import numpy as np
from torch.utils.data import Dataset
import scipy.sparse as sp
from proj_utils.graph import sparse_scipy2torch, get_laplacian


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
    idx, batch_input_ids, batch_masks, batch_tag_ids, batch_inputs = zip(*batch)

    batch_input_ids = t.from_numpy(np.array(batch_input_ids)).long()
    batch_masks = t.from_numpy(np.array(batch_masks)).float()
    batch_tag_ids = t.from_numpy(np.array(batch_tag_ids)).long()

    return  idx, batch_input_ids, batch_masks, batch_tag_ids, batch_inputs

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