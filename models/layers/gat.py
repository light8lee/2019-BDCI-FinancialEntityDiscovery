import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sp
import math


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_head, activation=None, residual=False, last_layer=False):
        super(GATLayer, self).__init__()
        assert out_dim % num_head == 0
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_head = num_head
        self.support_dim = out_dim // num_head
        self.activation = activation if activation is not None else torch.relu
        self.residual = residual

        self.weight = nn.Parameter(torch.FloatTensor(self.in_dim, self.out_dim))
        self.fc1 = nn.Parameter(torch.FloatTensor(1, 1, self.num_head, self.support_dim))
        self.fc2 = nn.Parameter(torch.FloatTensor(1, 1, self.num_head, self.support_dim))

        if self.residual:
            self.res_weight = nn.Parameter(torch.FloatTensor(self.in_dim, self.out_dim))

        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            std_div = 1. / math.sqrt(weight.size(-1))
            weight.data.uniform_(-std_div, std_div)

    def forward(self, A, X):
        head_outputs = []
        mask = (-1e9 * (1.0 - A)).unsqueeze(1)  # [b, 1, t, t]

        inputs = torch.matmul(X, self.weight)  # [b, t, h]
        shape = inputs.shape
        inputs = inputs.view(shape[0], shape[1], self.num_head, self.support_dim)  # [b, t, n, h/n]

        self_fc = self.fc1 * inputs  # [b, t, n, h/n]
        self_fc = torch.sum(self_fc, -1, keepdim=True).transpose(1, 2)  # [b, t, n, 1] -> [b, n, t, 1]
        neigh_fc = self.fc2 * inputs  # [b, t, n, h/n]
        neigh_fc = torch.sum(neigh_fc, -1, keepdim=True).permute(0, 2, 3, 1)  # [b, t, n, 1] -> [b, n, 1, t]

        alphas = F.leaky_relu(self_fc + neigh_fc)  # [b, n, t, t]
        alphas = alphas + mask  # only need neighborhoods' information
        alphas = torch.softmax(alphas, -1)  # [b, n, t, t]
        outputs = torch.matmul(alphas, inputs.transpose(1, 2))  # [b, n, t, h/n]

        outputs = outputs.transpose(1, 2).contiguous().view(*shape)  # [b, t, h]

        if self.residual:
            outputs += torch.matmul(X, self.res_weight)

        outputs = self.activation(outputs)

        return outputs

