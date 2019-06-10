import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sp
import math
from .layers.diffpool import DiffPool
from .layers import activation as Act
from .layers.pooling import MaxPooling, AvgPooling, SumPooling


class GNN_DiffPool_Base(nn.Module):
    def __init__(self, vocab_size, max_seq_len, drop_rate,
                 embedding_dim, num_diffpool_layer, diffpool_gnn,
                 pred_dims:list, ratio:float, readout_pool:str='max',
                 init_weight=None, activation=None, pred_act:str='ELU',
                 freeze:bool=False, **kwargs):
        super(GNN_DiffPool_Base, self).__init__()
        assert 0 < ratio <= 1.0
        pred_act = getattr(Act, pred_act, nn.ELU)

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.drop_rate = drop_rate
        self.embedding_dim = embedding_dim
        self.num_diffpool_layer = num_diffpool_layer
        self.diffpool_gnn = diffpool_gnn
        self.pred_dims = pred_dims
        self.ratio = ratio
        self.activation = activation

        if readout_pool == 'max':
            self.readout_pool = MaxPooling()
        elif readout_pool == 'avg':
            self.readout_pool = AvgPooling()
        elif readout_pool == 'sum':
            self.readout_pool = SumPooling()
        else:
            raise ValueError()

        self.embedding = self.init_unit_embedding(init_weight=init_weight)
        self.gnn_layers = nn.ModuleList()
        self.diffpool_layers = nn.ModuleList()
        self.dense = nn.Sequential()  # should be replaced in sub-class

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
        inputs = self.embedding(input_ids)
        concat_outputs = [inputs]
        for layer in self.gnn_layers:
            outputs = F.dropout(outputs, p=self.drop_rate, training=self.training)
            outputs = layer(input_adjs, outputs)
            concat_outputs.append(outputs)
        outputs = torch.cat(concat_outputs, -1)

        # TODO: add layer norm

        adjs = input_adjs
        pooled_outputs= []
        for layer in self.diffpool_layers:
            outputs = F.dropout(outputs, p=self.drop_rate, training=self.training)
            adjs, outputs = layer(adjs, outputs)
            pooled_outputs.append(
                self.readout_pool(outputs, 1)
            )

        # TODO: add subtract

        pooled_outputs = torch.cat(pooled_outputs, -1)
        outputs = self.dense(pooled_outputs)

        outputs = torch.log_softmax(outputs, 1)

        return outputs
