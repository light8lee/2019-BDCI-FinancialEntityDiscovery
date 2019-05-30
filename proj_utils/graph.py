from collections import Counter
from collections import defaultdict
import json
import networkx as nx
import math
import torch
import numpy as np
import scipy.sparse as sp
from .strings import is_enword

def get_edge_weight_by_offset(offset):
    if offset <= 1:
        return 1.
    elif offset <= 2:
        return 0.5
    elif offset <= 3:
        return 0.25
    elif offset <= 4:
        return 0.125
    else:
        return 0.05

def sparse_scipy2torch(sparse_matrix):
    if not sp.isspmatrix_coo(sparse_matrix):
        sparse_matrix = sp.coo_matrix(sparse_matrix)
    indices = np.vstack((sparse_matrix.row, sparse_matrix.col))
    values = sparse_matrix.data

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)

    shape = torch.Size(sparse_matrix.shape)
    return torch.sparse.FloatTensor(i, v, shape)


def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def get_laplacian(adj):
    adj_norm = normalize_adj(adj+sp.eye(adj.shape[0]))
    return adj_norm
