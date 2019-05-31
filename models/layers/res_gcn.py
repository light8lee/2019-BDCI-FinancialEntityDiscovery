import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sp
import math


class ResGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, activation=None):
        super(ResGCNLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = activation if activation is not None else torch.relu

        self.weight1 = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        self.weight2 = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        self.bias = nn.Parameter(torch.FloatTensor(out_dim))

        self.reset_parameters()

    def reset_parameters(self):
        std_div = 1. / math.sqrt(self.weight.size(-1))
        self.weight1.data.uniform_(-std_div, std_div)
        self.weight2.data.uniform_(-std_div, std_div)
        self.bias.data.uniform_(-std_div, std_div)

    def forward(self, A, X):
        """[summary]
        
        Arguments:
            A [b, 2t, 2t] -- [description]
            X [b, 2t, h1] -- [description]
        
        Returns:
            [b, 2t, h2] -- [description]
        """
        output = torch.matmul(X, self.weight1)  # [b, 2t, h2]
        output = torch.bmm(A, output) + self.bias  # [b, 2t, h2]
        skip_conn = torch.matmul(X, self.weight2)
        output = output + skip_conn
        output = self.activation(outout)
        return output
