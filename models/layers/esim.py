import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sp
import math

class ESimLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        assert out_dim % 2 == 0
        super(ESimLayer, self).__init__()
        self.rnn1 = nn.LSTM(in_dim, out_dim//2, 1, bidirectiona=True, batch_first=True)
        self.rnn2 = nn.LSTM(out_dim*4, out_dim//2, 1, bidirectiona=True, batch_first=True)
    
    def forward(self, inputs_a, inputs_b):
        outputs_a = self.rnn1(inputs_a)  # [b, t1, h]
        outputs_b = self.rnn1(inputs_b)  # [b, t2, h]

        attn = torch.bmm(outputs_a, outputs_b.transpose(-1, -2))  # [b, t1, t2]
        attn_a = torch.softmax(attn, -1)  # [b, t1, t2]
        attn_a = torch.matmul(attn_a, outputs_b)  # [b, t1, h]
        attn_b = torch.softmax(attn.transpose(-1, -2), -1)  # [b, t2, t1]
        attn_b = torch.matmul(attn_b, outputs_a)  # [b, t2, h]

        ma = torch.cat([outputs_a, attn_a, outputs_a-attn_a, outputs_a*attn_a], -1)  # [b, t1, 4h]
        mb = torch.cat([outputs_b, attn_b, outputs_b-attn_b, outputs_b*attn_b], -1)  # [b, t1, 4h]

        outputs_a = self.rnn2(ma)
        outputs_b = self.rnn2(mb)
        return outputs_a, outputs_b