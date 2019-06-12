import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sp
import math
from .layers.diffpool import DiffPool
from .layers import activation as Act
from .layers.pooling import MaxPooling, AvgPooling, SumPooling

class RNN_DiffPool(nn.Module):
    def __init__(self, vocab_size, max_seq_len, drop_rate, rnn,
                 embedding_dim, num_gnn_layer, gnn, 
                 pred_dims:list, rnn_dims:list, ratio:float, readout_pool:str='max',
                 init_weight=None, activation=None, pred_act:str='ELU',
                 freeze:bool=False, mode:str='add_norm', **kwargs):

        super(RNN_DiffPool, self).__init__()
        assert len(rnn_dims) >= 1
        assert 0 < ratio <= 1.0
        pred_act = getattr(Act, pred_act, nn.ELU)

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.drop_rate = drop_rate
        self.embedding_dim = embedding_dim
        self.pred_dims = pred_dims
        self.rnn_dims = rnn_dims
        self.freeze = freeze
        self.activation = activation
        self.mode = mode

        self.embedding = self.init_unit_embedding(init_weight=init_weight)
        self.rnn_layers = nn.ModuleList()
        self.diffpool_layers = nn.ModuleList()
        if readout_pool == 'max':
            self.readout_pool = MaxPooling()
        elif readout_pool == 'avg':
            self.readout_pool = AvgPooling()
        elif readout_pool == 'sum':
            self.readout_pool = SumPooling()
        else:
            raise ValueError()
        
        if rnn == 'GRU':
            rnn = nn.GRU
        elif rnn == 'LSTM':
            rnn = nn.LSTM
        else:
            raise ValueError()

        in_dim = self.embedding_dim
        flat_in_dim = self.embedding_dim
        for rnn_dim in self.rnn_dims:
            self.rnn_layers.append(
                rnn(in_dim, rnn_dim, num_layers=1, batch_first=True, bidirectional=True)
            )
            in_dim = rnn_dim * 2
            flat_in_dim += in_dim

        if self.mode == 'add_norm':
            self.norm = nn.LayerNorm(in_dim)
            self.res_weight = nn.Linear(self.embedding_dim, in_dim)
        elif self.mode == 'concat':
            in_dim = flat_in_dim
            self.norm = nn.LayerNorm(flat_in_dim)
        elif self.mode == 'origin':
            self.norm = nn.LayerNorm(in_dim)
        else:
            raise ValueError()

        out_dim = 0
        in_size = max_seq_len
        for _ in range(num_gnn_layer):
            self.diffpool_layers.append(
                DiffPool(in_dim, in_size, ratio, gnn, activation, **kwargs)
            )
            in_size = int(in_size * ratio)
            out_dim += in_dim

        out_dim += in_dim * (num_gnn_layer - 1)
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

    def forward(self, input_ids, input_masks, input_adjs):
        """[summary]
        
        Arguments:
            input_ids [2b, t] -- [description]
            input_masks [2b, t] -- [description]
            input_adjs [b, 2t, 2t] -- [description]
        
        Returns:
            [type] -- [description]
        """
        inputs = self.embedding(input_ids)  # [2b, t, e]
        outputs = inputs  # [2b, t, e]
        if self.mode == 'concat':
            flat_outputs = [outputs]

        for rnn in self.rnn_layers:
            outputs = F.dropout(outputs, p=self.drop_rate, training=self.training)
            outputs, _ = rnn(outputs)  # [2b, t, h]
            if self.mode == 'concat':
                flat_outputs.append(outputs)

        if self.mode == 'concat':
            outputs = torch.cat(flat_outputs, -1)  # [2b, t, h+e]
        hidden_dim = outputs.shape[-1]
        outputs = outputs * input_masks.unsqueeze(-1)  # [2b, t, h]
        outputs = outputs.contiguous().view(-1, 2*self.max_seq_len, hidden_dim)  # [b, 2t, h]
        if self.mode == 'add_norm':
            outputs = F.dropout(outputs, p=self.drop_rate, training=self.training)
            outputs = outputs + self.res_weight(inputs.view(-1, 2*self.max_seq_len, self.embedding_dim))
        outputs = self.norm(outputs)

        pooled_outputs = []
        adjs = input_adjs
        for layer in self.diffpool_layers:
            outputs = F.dropout(outputs, p=self.drop_rate, training=self.training)
            adjs, outputs = layer(adjs, outputs)  # [b, 2t, h2]
            pooled_output = self.readout_pool(outputs, 1)
            pooled_outputs.append(pooled_output)
        i = len(pooled_outputs) - 1
        while i > 0:
            pooled_outputs.append(pooled_outputs[i] - pooled_outputs[i-1])
            i -= 1
        pooled_outputs = torch.cat(pooled_outputs, -1)  # [b, h+...]

        outputs = self.dense(pooled_outputs)  # [b, 2]
        outputs = torch.log_softmax(outputs, 1)

        return outputs
