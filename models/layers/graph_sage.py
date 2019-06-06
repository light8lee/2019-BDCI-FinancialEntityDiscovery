import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sp
from .pooling import GlobalMaxPooling, GlobalAvgPooling, GlobalSumPooling
import math


class SAGELayer(nn.Module):
    def __init__(self, in_dim, out_dim, activation=None, pooling='max'):
        super(SAGELayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = activation if activation is not None else torch.relu
        if pooling == 'max':
            self.pooling = GlobalMaxPooling()
        elif pooling == 'mean':
            self.pooling = GlobalAvgPooling()
        elif pooling == 'sum':
            self.pooling = GlobalSumPooling()
        else:
            raise ValueError()

        self.weight = nn.Linear(self.in_dim, self.out_dim)
        self.fc = nn.Linear(self.in_dim+self.out_dim, self.out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            std_div = 1. / math.sqrt(weight.size(-1))
            weight.data.uniform_(-std_div, std_div)

    def forward(self, A, X):
        inputs = self.activation(self.weight(X))  # [b, t, h1]
        inputs = inputs.unsqueeze(1)  # [b, 1, t, h1]
        A = A.unsqueeze(-1)  # [b, t, t, 1]
        outputs = A * inputs  # [b, t, t, h1]
        outputs = self.pooling(outputs, -1)  # [b, t, h1]
        outputs = torch.cat([X, outputs], -1)  # [b, t, h0+h1]
        outputs = self.fc(outputs)  # [b, t, h]
        outputs = self.activation(outputs)  # [b, t, h]
        outputs = outputs / torch.norm(outputs, p=2)
        return outputs