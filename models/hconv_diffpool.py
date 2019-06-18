import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sp
import math
from .layers.diffpool import DiffPool
from .layers import activation as Act
from .layers.pooling import MaxPooling, AvgPooling, SumPooling
from .layers.hconv import HConvLayer


class HConv_DiffPool(nn.Module):
    def __init__(self, vocab_size, max_seq_len, drop_rate, hconv_gnn, diffpool_gnn,
                 embedding_dim, window_sizes, dilations, pre_cnn_dims, pre_gnn_dims, num_diffpool_layer, ratio,
                 pred_dims, readout_pool:str='sum', init_weight=None, need_embed:bool=False,
                 activation=None, pred_act:str='ELU', mode='concat', freeze:bool=False, **kwargs):
        super(HConv_DiffPool, self).__init__()
        assert len(dilations) == len(window_sizes) == len(pre_cnn_dims) == len(pre_gnn_dims)
        for dilation, window_size in zip(dilations, window_sizes):
            assert (dilation * (window_size - 1)) % 2 == 0
        assert 0 < ratio <= 1.0
        pred_act = getattr(Act, pred_act, nn.ELU)

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.drop_rate = drop_rate
        self.embedding_dim = embedding_dim
        self.pred_dims = pred_dims
        self.activation = activation
        self.mode = mode
        self.need_embed = need_embed
        self.freeze = freeze

        self.embedding = self.init_unit_embedding(init_weight=init_weight)
        if readout_pool == 'max':
            self.readout_pool = MaxPooling()
        elif readout_pool == 'avg':
            self.readout_pool = AvgPooling()
        elif readout_pool == 'sum':
            self.readout_pool = SumPooling()
        else:
            raise ValueError()
        
        self.pre_hconv_layers = nn.ModuleList()
        self.post_hconv_layers = nn.ModuleList()
        self.diffpool_layers = nn.ModuleList()

        in_dim = self.embedding_dim
        flat_in_dim = in_dim if need_embed else 0
        for pre_cnn_dim, pre_gnn_dim, dilation, window_size in zip(pre_cnn_dims, pre_gnn_dims, dilations, window_sizes):
            self.pre_hconv_layers.append(
                HConvLayer(in_dim, pre_cnn_dim, pre_gnn_dim, window_size, dilation, hconv_gnn,
                           activation=self.activation, **kwargs)
            )
            in_dim = pre_cnn_dim + pre_gnn_dim
            flat_in_dim += in_dim
        
        if self.mode == 'add_norm':
            self.norm = nn.LayerNorm(in_dim)
            self.res_weight = nn.Linear(self.embedding_dim, in_dim)
        elif self.mode == 'concat':
            in_dim = flat_in_dim
            self.norm = nn.LayerNorm(in_dim)
        elif self.mode == 'origin':
            self.norm = nn.LayerNorm(in_dim)
        else:
            raise ValueError()
        
        out_dim = 0
        for _ in range(num_diffpool_layer):
            self.diffpool_layers.append(
                DiffPool(in_dim, max_seq_len, ratio, diffpool_gnn,
                         activation=activation, **kwargs)
            )
            out_dim += in_dim
        out_dim += in_dim * (num_diffpool_layer - 1) 
       
        self.concat_norm = nn.LayerNorm(out_dim)
        pred_layers = []
        for pred_dim in pred_dims:
            pred_layers.append(
                nn.Linear(out_dim, pred_dim)
            )
            pred_layers.append(
                nn.LayerNorm(pred_dim)
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
        inputs = self.embedding(input_ids)
        outputs = inputs
        outputs = F.dropout(outputs, p=self.drop_rate, training=self.training)

        if self.mode == 'concat':
            flat_outputs = [outputs] if self.need_embed else []
        for layer in self.pre_hconv_layers:
            outputs = layer(input_adjs, outputs)
            if self.mode == 'concat':
                flat_outputs.append(outputs)
        
        if self.mode == 'concat':
            outputs = torch.cat(flat_outputs, -1)  # [2b, t, h]
        hidden_dim = outputs.shape[-1]
        outputs = outputs.view(-1, 2*self.max_seq_len, hidden_dim)  # [b, 2t, h]
        outputs = F.dropout(outputs, p=self.drop_rate, training=self.training)
        if self.mode == 'add_norm':
            outputs = F.dropout(outputs, p=self.drop_rate, training=self.training)
            outputs = outputs + self.res_weight(inputs)
        outputs = self.norm(outputs)

        pooled_outputs = []
        adjs = input_adjs
        for layer in self.diffpool_layers:
            adjs, outputs = layer(adjs, outputs)
            pooled_output = self.readout_pool(outputs, 1)
            pooled_outputs.append(pooled_output)

        i = len(pooled_outputs) - 1
        while i > 0:
            pooled_outputs.append(pooled_outputs[i] - pooled_outputs[i-1])
            i -= 1
        
        pooled_outputs = torch.cat(pooled_outputs, -1)  # [b, h]
        pooled_outputs = self.concat_norm(pooled_outputs)

        outputs = self.dense(pooled_outputs)  # [b, 2]
        outputs = torch.log_softmax(outputs, 1)

        return outputs