import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sp
import math
from .gcn import GCNLayer
from .gat import GATLayer
from .graph_sage import SAGELayer


class DiffPool(nn.Module):
    def __init__(self, in_dim, in_size, ratio, gnn='gcn', activation=None, **kwargs):
        super(DiffPool, self).__init__()
        if gnn == 'gcn':
            self.gnn_embed = GCNLayer(in_dim, in_dim, activation=activation,
                                      residual=kwargs['residual'])
            self.gnn_pool = GCNLayer(in_dim, int(in_size*ratio), activation=activation,
                                     residual=kwargs['residual'])
        elif gnn == 'gat':
            self.gnn_embed = GATLayer(in_dim, in_dim, kwargs['num_head'],
                                      activation=activation, residual=kwargs['residual'],
                                      last_layer=False)
            self.gnn_pool = GATLayer(in_dim, int(in_size*ratio), kwargs['num_head'],
                                     activation=activation, residual=kwargs['residual'],
                                     last_layer=False)
        elif gnn == 'sage':
            self.gnn_embed = SAGELayer(in_dim, in_dim, activation=activation,
                                       pooling=kwargs['pooling'])
            self.gnn_pool = SAGELayer(in_dim, int(in_size*ratio), activation=activation,
                                      pooling=kwargs['pooling'])
        else:
            raise ValueError()

        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            std_div = 1. / math.sqrt(weight.size(-1))
            weight.data.uniform_(-std_div, std_div)

    def forward(self, A, X):
        Z = self.gnn_embed(A, X)  # [b, t1, h]
        S = self.gnn_pool(A, X)  # [b, t1, t2]
        S = torch.softmax(S, -1)
        X_next = torch.bmm(S.transpose(-1, -2), Z)  # [b, t2, h]
        A_next = S.transpose(-1, -2).bmm(A).bmm(S)  # [b, t2, t2]
        return A_next, X_next
