import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sp
from .pooling import MaxPooling, AvgPooling, SumPooling
import math


class SAGELayer(nn.Module):
    def __init__(self, in_dim, out_dim, activation=None, pooling='avg'):
        super(SAGELayer, self).__init__()
        assert pooling in ['avg', 'max']
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = activation if activation is not None else torch.relu
        self.pooling = pooling
        # if pooling == 'max':
        #     self.pooling = MaxPooling()
        # elif pooling == 'mean':
        #     self.pooling = AvgPooling()
        # elif pooling == 'sum':
        #     self.pooling = SumPooling()
        # else:
        #     raise ValueError()

        self.weight = nn.Linear(self.in_dim, self.out_dim)
        self.fc = nn.Linear(self.in_dim+self.out_dim, self.out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            std_div = 1. / math.sqrt(weight.size(-1))
            weight.data.uniform_(-std_div, std_div)
    
    def avg_pool(self, A, outputs):
        pool = SumPooling()
        outputs = pool(outputs, 2)
        size = pool(A, 2)
        size += 1e-8
        outputs = outputs / size
        # outputs = outputs.masked_fill(size==0, 0)
        return outputs
    
    def max_pool(self, A, outputs):
        pool = MaxPooling()
        return pool(outputs, 2)

    def forward(self, A, X):
        inputs = self.activation(self.weight(X))  # [b, t, h1]
        inputs = inputs.unsqueeze(1)  # [b, 1, t, h1]
        A = A.unsqueeze(-1)  # [b, t, t, 1]
        outputs = A * inputs  # [b, t, t, h1]
        # outputs = self.pooling(outputs, 2)  # [b, t, h1] FIXME
        if self.pooling == 'avg':
            outputs = self.avg_pool(A, outputs)  # [b, t, h1]
        else:
            outputs = self.max_pool(A, outputs)
        outputs = torch.cat([X, outputs], -1)  # [b, t, h0+h1]
        outputs = self.fc(outputs)  # [b, t, h]
        outputs = self.activation(outputs)  # [b, t, h]
        outputs += 1e-8
        outputs = outputs / (torch.norm(outputs, p=2) + 1e-8)
        return outputs