import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sp
import math
from .layers.gcn import GCNLayer
from .layers import activation as Act
from .layers.pooling import GlobalMaxPooling

class CNN_GCN(nn.Module):
    def __init__(self, vocab_size, max_seq_len, drop_rate,
                 embedding_dim, window_size, hidden_dims:list, pred_dims:list,
                 init_weight=None, activation=None, pred_act:str='ELU',
                 residual:bool=False, freeze:bool=False, **kwargs):

        super(CNN_GCN, self).__init__()
        assert window_size % 2 == 1
        pred_act = getattr(Act, pred_act, nn.ELU)

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.drop_rate = drop_rate
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.hidden_dims = hidden_dims
        self.pred_dims = pred_dims
        self.freeze = freeze

        self.embedding = self.init_unit_embedding(init_weight=init_weight)
        self.conv1d = nn.Conv1d(self.embedding_dim, self.hidden_dims[0],
                                kernel_size=window_size, stride=1, padding=window_size//2)
        self.gcn_layers = nn.ModuleList()

        self.max_pool = GlobalMaxPooling()

        out_dim = 0
        for i in range(len(hidden_dims)-1):
            self.gcn_layers.append(
                GCNLayer(self.hidden_dims[i], self.hidden_dims[i+1], activation, residual)
            )
            out_dim += hidden_dims[i+1]

        pred_layers = []
        for pred_dim in pred_dims:
            pred_layers.append(
                nn.Linear(out_dim, pred_dim)
            )
            pred_layers.append(pred_act())
            out_dim = pred_dim
        pred_layers.append(
            nn.Linear(out_dim, 2)
        )

        self.dense = nn.Sequential(*pred_layers)

    def init_unit_embedding(self, init_weight=None, padding_idx=0):
        vecs = nn.Embedding(self.vocab_size, self.embedding_dim,
                            padding_idx=padding_idx)
        if init_weight is None:
            vecs.weight = nn.Parameter(
                torch.cat([
                    torch.zeros(1, self.embedding_dim),  # [PAD]
                    torch.FloatTensor(self.vocab_size-1,
                                      self.embedding_dim).uniform_(-0.5 / self.embedding_dim,
                                                                   0.5/self.embedding_dim)
                ])
            )
        else:
            vecs.weight = nn.Parameter(init_weight)
        vecs.weight.requires_grad = not self.freeze
        return vecs

    def forward(self, input_ids, input_masks, input_laps):
        """[summary]
        
        Arguments:
            input_ids [2b, t] -- [description]
            input_masks [2b, t] -- [description]
            input_laps [b, 2t, 2t] -- [description]
        
        Returns:
            [type] -- [description]
        """
        outputs = self.embedding(input_ids)  # [2b, t, e]
        outputs = outputs.transpose(-1, -2)  # [2b, e, t]
        outputs = self.conv1d(outputs).transpose(-1, -2)  # [2b, t, h1]
        outputs = outputs * input_masks.unsqueeze(-1)  # [2b, t, h1]
        outputs = outputs.view(-1, 2*self.max_seq_len, self.hidden_dims[0])  # [b, 2t, h1]

        pooled_outputs = []
        for layer in self.gcn_layers:
            outputs = F.dropout(outputs, p=self.drop_rate, training=self.training)
            outputs = layer(input_laps, outputs)  # [b, 2t, h2]
            pooled_output = self.max_pool(outputs)
            pooled_outputs.append(pooled_output)
        pooled_outputs = torch.cat(pooled_outputs, -1)  # [b, num_gcn_layer*h2]

        outputs = self.dense(pooled_outputs)  # [b, 2]
        outputs = torch.log_softmax(outputs, 1)

        return outputs
