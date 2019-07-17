import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sp
import math
from itertools import zip_longest
from .layers.gat import GATLayer
from .layers.gcn import GCNLayer
from .layers import activation as Act
from .layers.pooling import MaxPooling, AvgPooling, SumPooling

class RNN(nn.Module):
    def __init__(self, vocab_size, max_seq_len, drop_rate, readout_pool,
                 embedding_dim, gnn_hidden_dims, rnn_hidden_dims, rnn,
                 activation, residual, need_norm,
                 gnn_channels:list=None, init_weight=None, freeze:bool=False,
                 pred_dims:list=None, pred_act:str='ELU', **kwargs):
        super(RNN, self).__init__()
        for rnn_hidden_dim in rnn_hidden_dims:
            assert rnn_hidden_dim % 2 == 0
        gnn_channels = gnn_channels if gnn_channels else []
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
        self.gnn_channels = gnn_channels

        self.embedding = self.init_unit_embedding(init_weight=init_weight)
        self.rnn_layers = nn.ModuleList()  # to be replaced in subclass
        self.outer_gat_layers = nn.ModuleList()
        self.outer_gcn_layers = nn.ModuleList()

        if readout_pool == 'max':
            self.readout_pool = MaxPooling()
        elif readout_pool == 'avg':
            self.readout_pool = AvgPooling()
        elif readout_pool == 'sum':
            self.readout_pool = SumPooling()
        else:
            raise ValueError()

        if "gat" in gnn_channels:
            num_heads = kwargs['num_heads']
            assert len(num_heads) == len(rnn_hidden_dims)
        else:
            num_heads = []
        in_dim = embedding_dim
        for num_head, gnn_hidden_dim, rnn_hidden_dim in zip_longest(num_heads, gnn_hidden_dims, rnn_hidden_dims):
            out_dim = embedding_dim
            if "gat" in gnn_channels:
                self.outer_gat_layers.append(
                    GATLayer(in_dim, gnn_hidden_dim, num_head, self.activation, residual=residual, last_layer=False)
                )
                out_dim += gnn_hidden_dim
            if "gcn" in gnn_channels:
                self.outer_gcn_layers.append(
                    GCNLayer(in_dim, gnn_hidden_dim, self.activation, residual=residual)
                )
                out_dim += gnn_hidden_dim
            if rnn == 'lstm':
                self.rnn_layers.append(nn.LSTM(out_dim, rnn_hidden_dim//2, 1,
                                    bidirectional=True, batch_first=True))
            elif rnn == 'gru':
                self.rnn_layers.append(nn.GRU(out_dim, rnn_hidden_dim//2, 1,
                                    bidirectional=True, batch_first=True))
            in_dim = rnn_hidden_dim

        pred_act = getattr(Act, pred_act, nn.ELU)

        self.pred_dims = pred_dims
        pred_layers = []
        out_dim = in_dim * 4
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
                    torch.FloatTensor(self.vocab_size-1, self.embedding_dim).uniform_(-0.5 / \
                                                                             self.embedding_dim, 0.5/self.embedding_dim)
                ])
            )
        else:
            vecs.weight = nn.Parameter(init_weight)
        vecs.weight.requires_grad = not self.freeze
        return vecs

    def _normalize_adjs(self, input_masks, input_adjs):
        selfloop = torch.eye(self.max_seq_len*2, device=input_adjs.device).unsqueeze(0)  # [1, t, t]
        selfloop = selfloop * input_masks.unsqueeze(-1)  # [b, t, t], keep padding values to 0
        input_adjs = input_adjs + selfloop  # [b, t, t]
        if self.need_norm:  # this is equvalient to D^{-1/2} A D^{-1/2}
            row_sum = input_adjs.sum(-1)  # [b, t]
            d_inv_sqrt = torch.pow(row_sum, -0.5)  # [b, t]
            d_inv_sqrt *= input_masks  # [b, t], keep padding values to 0
            normalization = d_inv_sqrt.unsqueeze(-1) * d_inv_sqrt.unsqueeze(1) # [b, t, t]
            norm_adjs = input_adjs * normalization
            input_adjs = norm_adjs.masked_fill(input_adjs==0, 0)
        return input_adjs

    def forward(self, input_ids, input_masks, input_adjs):
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
        input_adjs = self._normalize_adjs(input_masks, input_adjs)

        outputs_a = self.embedding(inputs_a)  # [b, t, e]
        outputs_b = self.embedding(inputs_b)  # [b, t, e]

        for i, rnn_layer in enumerate(self.rnn_layers):
            outputs = torch.cat([outputs_a, outputs_b], 1)  # [b, 2t, e]
            extra_a_inputs = [outputs_a]
            extra_b_inputs = [outputs_b]

            if "gat" in self.gnn_channels:
                gat_outputs = self.outer_gat_layers[i](input_adjs, outputs)  # [b, 2t, e]
                gat_a_outputs, gat_b_outputs = torch.chunk(gat_outputs, 2, 1)  # [b, t, e] * 2

                extra_a_inputs.append(gat_a_outputs * masks_a)
                extra_b_inputs.append(gat_b_outputs * masks_b)

            if "gcn" in self.gnn_channels:
                gcn_outputs = self.outer_gcn_layers[i](input_adjs, outputs)
                gcn_a_outputs, gcn_b_outputs = torch.chunk(gcn_outputs, 2, 1)  # [b, t, e] * 2

                extra_a_inputs.append(gcn_a_outputs * masks_a)
                extra_b_inputs.append(gcn_b_outputs * masks_b)
            inputs_a = torch.cat(extra_a_inputs, -1)  # [b, t, h]
            inputs_b = torch.cat(extra_b_inputs, -1)  # [b, t, h]

            outputs_a, _ = rnn_layer(inputs_a)  # [b, t, h]
            outputs_a = outputs_a * masks_a  # [b, t, h]

            outputs_b, _ = rnn_layer(inputs_b)  # [b, t, h]
            outputs_b = outputs_b * masks_b  # [b, t, h]

        pool_a = self.readout_pool(outputs_a, 1)  # [b, h]
        pool_b = self.readout_pool(outputs_b, 1)  # [b, h]

        outputs = torch.cat(
            [pool_a, pool_b, torch.abs(pool_a-pool_b), pool_a*pool_b],
            -1
        )
        outputs = self.dense(outputs)  # [b, 2]
        outputs = torch.log_softmax(outputs, 1)

        return outputs