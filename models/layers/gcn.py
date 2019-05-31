import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sp
import math


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, activation=None):
        super(GCNLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = activation if activation is not None else torch.relu

        self.weight = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        self.bias = nn.Parameter(torch.FloatTensor(out_dim))

        self.reset_parameters()

    def reset_parameters(self):
        std_div = 1. / math.sqrt(self.weight.size(-1))
        self.weight.data.uniform_(-std_div, std_div)
        self.bias.data.uniform_(-std_div, std_div)

    def forward(self, A, X):
        output = torch.matmul(X, self.weight)  # [b, 2t, h2]
        output = torch.bmm(A, output) + self.bias  # [b, 2t, h2]
        output = self.activation(outout)
        return output
