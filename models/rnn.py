import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sp
import math
from itertools import zip_longest
from .layers.diffpool import DiffPool
from .layers.graph_sage import SAGELayer
from .layers.gat import GATLayer
from .layers.gcn import GCNLayer
from .layers import activation as Act
from .layers.pooling import MaxPooling, AvgPooling, SumPooling
from .layers.normalization import normalize_adjs

class RNN(nn.Module):
    def __init__(self, vocab_size, max_seq_len, drop_rate, readout_pool,
                 embedding_dim, gnn_hidden_dims, rnn_hidden_dims, rnn,
                 activation, residual, need_norm, gnn,
                 init_weight=None, freeze:bool=False,
                 pred_dims:list=None, pred_act:str='ELU', **kwargs):
        super(RNN, self).__init__()
        rnn = rnn.lower()
        assert rnn in ['lstm', 'gru']
        assert gnn in ["diffpool", "gcn", "gat", "none"]
        for rnn_hidden_dim in rnn_hidden_dims:
            assert rnn_hidden_dim % 2 == 0
        pred_dims = pred_dims if pred_dims else []

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.drop_rate = drop_rate
        self.embedding_dim = embedding_dim
        self.gnn_hidden_dims = gnn_hidden_dims
        self.rnn_hidden_dims = rnn_hidden_dims
        self.activation = getattr(Act, activation)
        self.freeze = freeze
        self.need_norm = need_norm
        self.gnn = gnn

        self.embedding = self.init_unit_embedding(init_weight=init_weight)
        self.rnn_layers = nn.ModuleList()  # to be replaced in subclass
        self.gnn_layers = nn.ModuleList()

        if readout_pool == 'max':
            self.readout_pool = MaxPooling()
        elif readout_pool == 'avg':
            self.readout_pool = AvgPooling()
        elif readout_pool == 'sum':
            self.readout_pool = SumPooling()
        else:
            raise ValueError()

        in_dim = embedding_dim
        for rnn_hidden_dim in rnn_hidden_dims:
            # out_dim = embedding_dim
            if rnn == 'lstm':
                self.rnn_layers.append(nn.LSTM(in_dim, rnn_hidden_dim//2, 1,
                                    bidirectional=True, batch_first=True))
            elif rnn == 'gru':
                self.rnn_layers.append(nn.GRU(in_dim, rnn_hidden_dim//2, 1,
                                    bidirectional=True, batch_first=True))
            in_dim = rnn_hidden_dim

        out_dim = in_dim * 4
        if gnn != "none": 
            in_size = self.max_seq_len * 2
            for i, gnn_hidden_dim in enumerate(gnn_hidden_dims):
                if gnn == "diffpool":
                    self.gnn_layers.append(
                        DiffPool(in_dim, in_size, kwargs['ratio'],
                                gnn='gcn', activation=self.activation, residual=residual)
                    )
                    in_size = int(in_size * kwargs['ratio'])
                elif gnn == "gat":
                    num_head = kwargs['num_heads'][i]
                    self.gnn_layers.append(
                        GATLayer(in_dim, gnn_hidden_dim, num_head, self.activation, residual=residual, last_layer=False)
                    )
                elif gnn == "gcn":
                    self.gnn_layers.append(
                        GCNLayer(in_dim, gnn_hidden_dim, self.activation, residual=residual)
                    )
                in_dim = gnn_hidden_dim
                out_dim += in_dim

        pred_act = getattr(Act, pred_act, nn.ELU)

        self.pred_dims = pred_dims
        pred_layers = []
        for pred_dim in pred_dims:
            pred_layers.append(
                nn.Linear(out_dim, pred_dim)
            )
            pred_layers.append(pred_act())
            pred_layers.append(nn.Dropout(p=self.drop_rate))
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
                    torch.FloatTensor(self.vocab_size-1, self.embedding_dim).uniform_(-0.5 / \
                                                                             self.embedding_dim, 0.5/self.embedding_dim)
                ])
            )
        else:
            vecs.weight = nn.Parameter(init_weight)
        vecs.weight.requires_grad = not self.freeze
        return vecs

    def forward(self, input_ids, input_masks):
        """[summary]

        Arguments:
            input_ids [2b, t] -- [description]
            input_masks [2b, t] -- [description]
            input_adjs [b, 2t, 2t] -- [description]

        Returns:
            [type] -- [description]
        """
        inputs_a, inputs_b = input_ids
        masks_a, masks_b = input_masks
        masks_a = masks_a.unsqueeze(-1)
        masks_b = masks_b.unsqueeze(-1)

        input_masks = torch.cat(input_masks, 1)  # [b, 2t]

        outputs_a = self.embedding(inputs_a)  # [b, t, e]
        outputs_b = self.embedding(inputs_b)  # [b, t, e]

        for i, rnn_layer in enumerate(self.rnn_layers):
            outputs_a, _ = rnn_layer(outputs_a)  # [b, t, h]
            outputs_a = outputs_a * masks_a  # [b, t, h]

            outputs_b, _ = rnn_layer(outputs_b)  # [b, t, h]
            outputs_b = outputs_b * masks_b  # [b, t, h]

        pool_a = self.readout_pool(outputs_a, 1)  # [b, h]
        pool_b = self.readout_pool(outputs_b, 1)  # [b, h]

        sim_outputs = []
        pool_a = self.readout_pool(outputs_a, 1)
        pool_b = self.readout_pool(outputs_b, 1)
        # sim_outputs.append(self._cos_sim(pool_a, pool_b).unsqueeze(-1))
        sim_outputs.append(pool_a)
        sim_outputs.append(pool_b)
        sim_outputs.append(torch.abs(pool_a - pool_b))
        sim_outputs.append(pool_a * pool_b)

        outputs = torch.cat([outputs_a, outputs_b], 1)  # [b, 2t, e]
        for gnn_layer in self.gnn_layers:
            input_adjs = torch.bmm(outputs, outputs.transpose(-1, -2))  # [b, 2t, 2t]
            if self.need_norm:
                input_adjs = normalize_adjs(input_masks, input_adjs)
            if self.gnn == 'diffpool':
                input_adjs, outputs = gnn_layer(input_adjs, outputs)  # [b, 2t, e]
                sim_outputs.append(self.readout_pool(outputs, 1))
            else:
                outputs = gnn_layer(input_adjs, outputs)
                sim_outputs.append(self.readout_pool(outputs, 1))

        outputs = torch.cat(sim_outputs, -1)
        outputs = self.dense(outputs)  # [b, 1]
        outputs = torch.log_softmax(outputs, 1)

        return outputs