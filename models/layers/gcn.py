import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sp
import math


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, activation=None, residual=False):
        super(GCNLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = activation if activation is not None else torch.relu
        self.residual = residual

        self.weight = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        if self.residual:
            self.res_weight = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        self.bias = nn.Parameter(torch.FloatTensor(out_dim))

        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            std_div = 1. / math.sqrt(weight.size(-1))
            weight.data.uniform_(-std_div, std_div)

    def forward(self, A, X):
        outputs = torch.matmul(X, self.weight)  # [b, 2t, h2]
        outputs = torch.bmm(A, outputs) + self.bias  # [b, 2t, h2]
        if self.residual:
            outputs += torch.matmul(X, self.res_weight)  # [b, 2t, h2]
        outputs = self.activation(outputs)
        return outputs
