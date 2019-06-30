import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sp
import math


class ABCNN1(nn.Module):
    def __init__(self, in_dim, out_dim, max_seq_len, window_size, activation=None, num_channel=2):
        super(ABCNN1, self).__init__()
        assert window_size % 2 == 1
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = activation if activation is not None else torch.relu
        self.num_channel = num_channel

        self.weight = nn.Parameter(torch.FloatTensor(max_seq_len, in_dim))
        self.conv = nn.Conv2d(num_channel, out_dim, kernel_size=(window_size, in_dim),
                              padding=(window_size//2, 0), stride=1, dilation=1)
        
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            std_div = 1. / math.sqrt(weight.size(-1))
            weight.data.uniform_(-std_div, std_div)
        torch.nn.init.xavier_uniform_(self.conv.weight.data)
    
    def forward(self, xa, xb, extra_a_fea=None, extra_b_fea=None):
        extra_a_fea = [] if extra_a_fea is None else list(extra_a_fea)
        extra_b_fea = [] if extra_b_fea is None else list(extra_b_fea)
        assert len(extra_a_fea) == len(extra_b_fea) == (self.num_channel - 2)

        xa_expanded = xa.unsqueeze(-2)  # [b, t1, 1, e]
        xb_expanded = xb.unsqueeze(1)  # [b, 1, t2, e]

        attn = torch.pow(xa_expanded - xb_expanded, 2)  # [b, t1, t2, e]
        attn = torch.sum(attn, -1)  # [b, t1, t2]
        attn = 1 / (torch.sqrt(attn) + 1)  # [b, t1, t2]

        xa_attn = torch.matmul(attn, self.weight)  # [b, t1, e]
        xb_attn = torch.matmul(attn.transpose(-1, -2), self.weight)  # [b, t2, e]


        extra_a_fea.extend([xa, xa_attn])
        extra_b_fea.extend([xb, xb_attn])
        xa = torch.stack(extra_a_fea, 1)  # [b, c, t, e]
        xb = torch.stack(extra_b_fea, 1)  # [b, c, t, e]

        conv_a = self.conv(xa)  # [b, h, t, 1]
        conv_b = self.conv(xb)  # [b, h, t, 1]
        conv_a = conv_a.squeeze(-1).transpose(-1, -2)  # [b, t, h]
        conv_b = conv_b.squeeze(-1).transpose(-1, -2)  # [b, t, h]

        conv_a = self.activation(conv_a)
        conv_b = self.activation(conv_b)

        return conv_a, conv_b

        