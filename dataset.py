import torch as t
import pickle
import numpy as np
from torch.utils.data import Dataset
import scipy.sparse as sp
from proj_utils.graph import sparse_scipy2torch, get_laplacian


class GraphDataset(Dataset):
    def __init__(self, fea_file, targets, postions):
        self.fea_file = fea_file
        self.targets = targets
        self.postions = postions

    def __len__(self):
        return len(self.targets)
        # return 100

    def __getitem__(self, idx):
        pos = self.postions[idx]
        self.fea_file.seek(pos)
        feature = pickle.load(self.fea_file)
        target = self.targets[idx]
        return feature, target


def collect_multigraph(batch):
    batch_size = len(batch)
    features, targets = zip(*batch)
    batch_inputs_a, batch_mask_a, batch_inputs_b, batch_mask_b = zip(*features)
    seq_len = len(batch_inputs_a[0])

    batch_inputs_a = t.from_numpy(np.array(batch_inputs_a)).long()
    batch_inputs_b = t.from_numpy(np.array(batch_inputs_b)).long()
    batch_inputs = (batch_inputs_a, batch_inputs_b)

    batch_mask_a = t.from_numpy(np.array(batch_mask_a)).float()
    batch_mask_b = t.from_numpy(np.array(batch_mask_b)).float()
    batch_masks = (batch_mask_a, batch_mask_b)

    targets = t.from_numpy(np.array(targets))

    return (batch_inputs, batch_masks), targets


def collect_single(batch):
    batch_size = len(batch)
    features, targets = zip(*batch)
    idx, batch_inputs, batch_masks, batch_types = zip(*features)
    seq_len = len(batch_inputs[0])

    batch_inputs = t.from_numpy(np.array(batch_inputs)).long()
    batch_masks = t.from_numpy(np.array(batch_masks)).float()
    batch_types = t.from_numpy(np.array(batch_types)).long()

    targets = t.from_numpy(np.array(targets))

    return (idx, batch_inputs, batch_masks, batch_types), targets
