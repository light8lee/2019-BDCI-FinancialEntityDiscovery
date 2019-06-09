import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sp
import math


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, activation=None, residual=False, num_mlp_layer=1):
        super(GCNLayer, self).__init__()
        assert num_mlp_layer > 0
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = activation if activation is not None else torch.relu
        self.residual = residual
        self.num_mlp_layer = num_mlp_layer

        weight = nn.Linear(in_dim, out_dim)
        self.mlp_layers = nn.ModuleList([weight])
        num_mlp_layer -= 1
        if self.residual:
            self.res_layer = nn.Linear(in_dim, out_dim)
        for _ in range(num_mlp_layer):
            self.mlp_layers.append(
                nn.Linear(out_dim, out_dim)
            )


        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            std_div = 1. / math.sqrt(weight.size(-1))
            weight.data.uniform_(-std_div, std_div)

    def forward(self, A, X):
        outputs = torch.bmm(A, X) # [b, t, h1]
        outputs = self.mlp_layers[0](X)  # [b, t, h2]
        if self.residual:
            outputs += self.res_layer(X)  # [b, 2t, h2]
        outputs = self.activation(outputs)
        for layer in self.mlp_layers[1:]:
            outputs = self.activation(outputs)
        return outputs
