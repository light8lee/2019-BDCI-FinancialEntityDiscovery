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
        self.last_layer = last_layer

        self.weights = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(self.in_dim, self.support_dim)) for _ in range(self.num_head)
        ])
        self.fc1s = nn.ModuleList([
            nn.Linear(self.support_dim, 1) for _ in range(self.num_head)
        ])
        self.fc2s = nn.ModuleList([
            nn.Linear(self.support_dim, 1) for _ in range(self.num_head)
        ])

        if self.residual:
            self.res_weights = nn.ParameterList([
                nn.Parameter(torch.FloatTensor(self.in_dim, self.support_dim)) for _ in range(self.num_head)
            ])

        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            std_div = 1. / math.sqrt(weight.size(-1))
            weight.data.uniform_(-std_div, std_div)

    def forward(self, A, X):
        head_outputs = []
        mask = (-1e9 * (1.0 - A))
        for i in range(self.num_head):
            inputs = torch.matmul(X, self.weights[i])  # [b, t, h]
            self_fc = self.fc1s[i](inputs)  # [b, t, 1]
            neigh_fc = self.fc2s[i](inputs).transpose(-1, -2)  # [b, t, 1]
            alphas = F.leaky_relu(self_fc + neigh_fc)  # [b, t, t]
            alphas = alphas + mask  # only need neighborhoods' information
            alphas = torch.softmax(alphas, -1)  # [b, t, t]
            outputs = torch.bmm(alphas, inputs)  # [b, t, h]
            if self.residual:
                outputs += torch.matmul(X, self.res_weights[i])
            head_outputs.append(self.activation(outputs))
        if self.last_layer:
            outputs = torch.stack(head_outputs)  # [n, b, t, h]
            outputs = torch.mean(outputs, 0)  # [b, t, h]
        else:
            outputs = torch.cat(head_outputs, -1)  # [b, t, h*n]
        return outputs

