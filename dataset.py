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
    batch_inputs, batch_masks, batch_tags = zip(*batch)

    batch_inputs = t.from_numpy(np.array(batch_inputs)).long()
    batch_masks = t.from_numpy(np.array(batch_masks)).float()
    batch_tags = t.from_numpy(np.array(batch_tags)).long()

    return  batch_inputs, batch_masks, batch_tags
