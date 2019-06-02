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

    def __getitem__(self, idx):
        pos = self.postions[idx]
        self.fea_file.seek(pos)
        feature = pickle.load(self.fea_file)
        target = self.targets[idx]
        return feature, target


def collect_multigraph(need_norm, batch):
    batch_size = len(batch)
    features, targets = zip(*batch)
    batch_inputs_a, batch_mask_a, batch_inputs_b, batch_mask_b, batch_rows, batch_cols = zip(*features)
    seq_len = len(batch_inputs_a[0])
    shape = (2*seq_len, 2*seq_len)

    batch_inputs = []
    for inputs_a, inputs_b in zip(batch_inputs_a, batch_inputs_b):
        batch_inputs.append(inputs_a)
        batch_inputs.append(inputs_b)

    batch_masks = []
    for mask_a, mask_b in zip(batch_mask_a, batch_mask_b):
        batch_masks.append(mask_a)
        batch_masks.append(mask_b)

    batch_inputs = t.from_numpy(np.array(batch_inputs)).long()
    batch_masks = t.from_numpy(np.array(batch_masks)).float()

    batch_adjs = []
    for rows, cols in zip(batch_rows, batch_cols):
        data = np.ones_like(rows)
        mtx = sp.coo_matrix((data, (rows, cols)), shape=shape)
        mtx = mtx.transpose() + mtx  # 下三角加上上三角构成完整的邻接矩阵
        if need_norm:
            mtx = get_laplacian(mtx)
        else:
            mtx = mtx + sp.eye(2*seq_len)
        mtx = sparse_scipy2torch(mtx)
        batch_adjs.append(mtx)
    batch_adjs = t.stack(batch_adjs, 0)
    batch_adjs = batch_adjs.to_dense().float()

    targets = t.from_numpy(np.array(targets)).long()

    return (batch_inputs, batch_masks, batch_adjs), targets


if __name__ == '__main__':
    fea_file = open('inputs/dev.fea', 'rb')
    with open('inputs/dev.tgt', 'r') as f:
        targets = [int(v.strip()) for v in f]
    with open('inputs/dev.pos', 'r') as f:
        positions = [int(v.strip()) for v in f]
    dataset = GraphDataset(fea_file, targets, positions)
    dl = t.utils.data.DataLoader(dataset, batch_size=2, collate_fn=collect_multigraph)
    for data in dl:
        print(data)
