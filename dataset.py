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
