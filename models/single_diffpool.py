import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sp
import math
from .layers.diffpool import DiffPool
from .layers.gat import GATLayer
from .layers.gcn import GCNLayer
from .layers import activation as Act
from .layers.pooling import MaxPooling, AvgPooling, SumPooling


class GNN_DiffPool_Base(nn.Module):
    def __init__(self, vocab_size, max_seq_len, drop_rate,
                 embedding_dim, num_diffpool_layer, diffpool_gnn, hidden_dims:list,
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
        self.hidden_dims = hidden_dims
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

        in_dim = sum(hidden_dims) + embedding_dim
        in_size = max_seq_len
        out_dim = 0
        for _ in range(num_diffpool_layer):
            self.diffpool_layers.append(
                DiffPool(in_dim, in_size, ratio, diffpool_gnn, activation, **kwargs)
            )
            in_size = int(in_size * ratio)
            out_dim += in_dim

        out_dim += in_dim * (num_diffpool_layer - 1)
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

        i = len(pooled_outputs) - 1
        while i > 0:
            pooled_outputs.append(pooled_outputs[i] - pooled_outputs[i-1])
            i -= 1

        pooled_outputs = torch.cat(pooled_outputs, -1)
        outputs = self.dense(pooled_outputs)

        outputs = torch.log_softmax(outputs, 1)

        return outputs


class GAT_DiffPool(GNN_DiffPool_Base):
    def __init__(self, vocab_size, max_seq_len, drop_rate,
                 embedding_dim, num_diffpool_layer, diffpool_gnn, hidden_dims:list,
                 pred_dims:list, ratio:float, readout_pool:str='max',
                 init_weight=None, activation=None, pred_act:str='ELU',
                 freeze:bool=False, num_heads:list=None, residual=False, **kwargs):
        super(GAT_DiffPool, self).__init__(vocab_size, max_seq_len, drop_rate,
                                           embedding_dim, num_diffpool_layer, diffpool_gnn,
                                           hidden_dims, pred_dims, ratio, readout_pool, init_weight,
                                           activation, pred_act, freeze, **kwargs)
        assert len(num_heads) == len(hidden_dims)
        self.num_heads = num_heads
        in_dim = self.embedding_dim
        for hidden_dim, num_head in zip(self.hidden_dims, self.num_heads):
            self.gnn_layers.append(
                GATLayer(in_dim, hidden_dim, num_head,
                         activation=self.activation, residual=residual, last_layer=False)
            )
            in_dim = hidden_dim


class GCN_DiffPool(GNN_DiffPool_Base):
    def __init__(self, vocab_size, max_seq_len, drop_rate,
                 embedding_dim, num_diffpool_layer, diffpool_gnn, hidden_dims:list,
                 pred_dims:list, ratio:float, readout_pool:str='max',
                 init_weight=None, activation=None, pred_act:str='ELU',
                 freeze:bool=False, residual=False, **kwargs):
        super(GCN_DiffPool, self).__init__(vocab_size, max_seq_len, drop_rate,
                                           embedding_dim, num_diffpool_layer, diffpool_gnn,
                                           hidden_dims, pred_dims, ratio, readout_pool, init_weight,
                                           activation, pred_act, freeze, **kwargs)
        in_dim = self.embedding_dim
        for hidden_dim in self.hidden_dims:
            self.gnn_layers.append(
                GCNLayer(in_dim, hidden_dim,
                         activation=self.activation, residual=residual)
            )
            in_dim = hidden_dim
