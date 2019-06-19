import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sp
import math


class ABCNN1(nn.Module):
    def __init__(self, in_dim, out_dim, max_seq_len, window_size, activation=None):
        super(ABCNN1, self).__init__()
        assert window_size % 2 == 1
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = activation if activation is not None else torch.relu

        self.weight = nn.Parameter(torch.FloatTensor(max_seq_len, in_dim))
        self.conv = nn.Conv2d(2, out_dim, kernel_size=(in_dim, window_size),
                              padding=(0, window_size//2), stride=1, dilation=1)
        
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            std_div = 1. / math.sqrt(weight.size(-1))
            weight.data.uniform_(-std_div, std_div)
    
    def forward(self, xa, xb):
        xa = xa.transpose(-1, -2)  # [b, e, t1]
        xb = xb.transpose(-1, -2)  # [b, e, t2]
        xa = xa.unsqueeze(-1)  # [b, e, t1, 1]
        xb = xb.unsqueeze(-1)  # [b, e, t2, 1]

        attn = torch.pow(xa - xb.transpose(-1, -2), 2)  # [b, e, t1, t2]
        attn = torch.sum(attn, 1)  # [b, t1, t2]
        attn = 1 / (torch.sqrt(attn) + 1)  # [b, t1, t2]

        xa_attn = torch.matmul(attn, self.weight)  # [b, t1, e]
        xb_attn = torch.matmul(attn.transpose(-1, -2), self.weight)  # [b, t2, e]

        xa_attn = xa_attn.transpose(-1, -2).unsqueeze(-1)  # [b, e, t1, 1]
        xb_attn = xb_attn.transpose(-1, -2).unsqueeze(-1)  # [b, e, t2, 1]

        xa = torch.cat([xa, xa_attn], axis=-1)  # [b, e, t, 2]
        xb = torch.cat([xb, xb_attn], axis=-1)  # [b, e, t, 2]

        conv_a = self.conv(xa).squeeze(1)  # [b, t, h]
        conv_b = self.conv(xb).squeeze(1)  # [b, t, h]

        conv_a = self.activation(conv_a)
        conv_b = self.activation(conv_b)

        return conv_a, conv_b

        