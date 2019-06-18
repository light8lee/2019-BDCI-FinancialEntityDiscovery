import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sp
import math
from .gcn import GCNLayer
from .gat import GATLayer
from .graph_sage import SAGELayer

class HConvLayer(nn.Module):
    def __init__(self, in_dim, out_cnn_dim, out_gnn_dim, window_size, dilation, gnn='gcn', activation=None, **kwargs):
        super(HConvLayer, self).__init__()
        assert (dilation * (window_size - 1)) % 2 == 0
        self.in_dim = in_dim
        self.out_cnn_dim = out_cnn_dim
        self.out_gnn_dim = out_gnn_dim
        self.activation = activation

        if gnn == 'gcn':
            self.gnn = GCNLayer(in_dim, out_gnn_dim, activation=activation,
                                residual=kwargs['residual'])
        elif gnn == 'gat':
            self.gnn = GATLayer(in_dim, out_gnn_dim, kwargs['num_head'],
                                activation=activation, residual=kwargs['residual'],
                                last_layer=False)
        elif gnn == 'sage':
            self.gnn = SAGELayer(in_dim, out_gnn_dim, activation=activation,
                                 pooling=kwargs['pooling'])
        else:
            raise ValueError()
        
        padding = (dilation * (window_size - 1)) // 2
        self.conv1d = nn.Conv1d(in_dim, out_cnn_dim, window_size,
                                stride=1, padding=padding, dilation=dilation)
    
    def forward(self, A, inputs):
        """[summary]
        
        Arguments:
            A [b, 2t, 2t] -- [description]
            inputs [2b, t, e] -- [description]
        """
        batch_size, seq_len, _ = inputs.shape
        conv_inputs = inputs.transpose(-1, -2)  # [2b, e, t]
        conv_outputs = self.conv1d(conv_inputs)  # [2b, h, t]
        conv_outputs = conv_outputs.transpose(-1, -2)  # [2b, t, h]
        conv_outputs = self.activation(conv_outputs)

        gnn_inputs = inputs.view(-1, 2*seq_len, self.in_dim)  # [b, 2t, e]
        gnn_outputs = self.gnn(A, gnn_inputs)  # [b, 2t, h]
        gnn_outputs = gnn_outputs.view(-1, seq_len, self.out_gnn_dim)  # [2b, t, h]
        outputs = torch.cat([conv_outputs, gnn_outputs], -1)  # [2b, t, 2h]

        return outputs
