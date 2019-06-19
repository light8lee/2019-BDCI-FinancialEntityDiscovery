import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sp
import math
from .layers.diffpool import DiffPool
from .layers import activation as Act
from .layers.abcnn import ABCNN1
from .layers.pooling import MaxPooling, AvgPooling, SumPooling

class ABCNN1_DiffPool(nn.Module):
    def __init__(self, vocab_size, max_seq_len, drop_rate,
                 embedding_dim, window_sizes, num_diffpool_layer, diffpool_gnn, 
                 pred_dims:list, cnn_dims:list, ratio:float, readout_pool:str='max',
                 init_weight=None, activation=None, pred_act:str='ELU',
                 freeze:bool=False, mode:str='add_norm', need_embed:bool=True, **kwargs):

        super(ABCNN1_DiffPool, self).__init__()
        assert len(window_sizes) == len(cnn_dims)
        assert 0 < ratio <= 1.0
        pred_act = getattr(Act, pred_act, nn.ELU)

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.drop_rate = drop_rate
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.pred_dims = pred_dims
        self.cnn_dims = cnn_dims
        self.freeze = freeze
        self.activation = activation
        self.mode = mode
        self.need_embed = need_embed

        self.embedding = self.init_unit_embedding(init_weight=init_weight)
        self.cnn_layers = nn.ModuleList()
        self.diffpool_layers = nn.ModuleList()
        if readout_pool == 'max':
            self.readout_pool = MaxPooling()
        elif readout_pool == 'avg':
            self.readout_pool = AvgPooling()
        elif readout_pool == 'sum':
            self.readout_pool = SumPooling()
        else:
            raise ValueError()

        in_dim = embedding_dim
        flat_in_dim = in_dim if need_embed else 0
        for cnn_dim, window_size in zip(cnn_dims, window_sizes):
            self.cnn_layers.append(
                ABCNN1(in_dim, cnn_dim, max_seq_len, window_size, activation=activation)
            )
            in_dim = cnn_dim
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
        self.concat_norm = nn.LayerNorm(out_dim)
        pred_layers = []
        for pred_dim in pred_dims:
            pred_layers.append(
                nn.Linear(out_dim, pred_dim)
            )
            pred_layers.append(
                nn.Dropout(p=self.drop_rate)
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
            input_ids ([b, t], [b, t]) -- [description]
            input_masks ([b, t], [b, t]) -- [description]
            input_adjs [b, 2t, 2t] -- [description]
        
        Returns:
            [type] -- [description]
        """
        inputs_a, inputs_b = input_ids
        masks_a, masks_b = input_masks
        inputs_a = self.embedding(inputs_a)  # [b, t, e]
        inputs_b = self.embedding(inputs_b)  # [b, t, e]

        outputs_a = F.dropout(inputs_a, p=self.drop_rate, training=self.training)
        outputs_b = F.dropout(inputs_b, p=self.drop_rate, training=self.training)

        if self.mode == 'concat':
            flat_outputs = [torch.cat([outputs_a, outputs_b], 1)] if self.need_embed else []

        for conv in self.cnn_layers:
            outputs_a, outputs_b = conv(outputs_a, outputs_b)  # [b, t, h]
            if self.mode == 'concat':
                flat_outputs.append(torch.cat([outputs_a, outputs_b], 1))

        if self.mode == 'concat':
            outputs = torch.cat(flat_outputs, -1)  # [b, 2t, h+e]
        else:
            outputs = torch.cat([outputs_a, outputs_b], 1)  # [b, 2t, h]

        hidden_dim = outputs.shape[-1]
        inputs_masks = torch.cat([masks_a, masks_b], 1)  # [b, 2t]
        outputs = outputs * input_masks.unsqueeze(-1)  # [b, 2t, h]
        outputs = outputs.contiguous().view(-1, 2*self.max_seq_len, hidden_dim)  # [b, 2t, h]

        if self.mode == 'add_norm':
            inputs = torch.cat([inputs_a, inputs_b], 1)
            outputs = outputs + self.res_weight(inputs.view(-1, 2*self.max_seq_len, self.embedding_dim))
        outputs = self.norm(outputs)

        pooled_outputs = []
        adjs = input_adjs
        for layer in self.diffpool_layers:
            adjs, outputs = layer(adjs, outputs)  # [b, 2t, h2]
            pooled_output = self.readout_pool(outputs, 1)
            pooled_outputs.append(pooled_output)
        i = len(pooled_outputs) - 1
        while i > 0:
            pooled_outputs.append(pooled_outputs[i] - pooled_outputs[i-1])
            i -= 1
        pooled_outputs = torch.cat(pooled_outputs, -1)  # [b, h+...]
        pooled_outputs = self.concat_norm(pooled_outputs)

        outputs = self.dense(pooled_outputs)  # [b, 2]
        outputs = torch.log_softmax(outputs, 1)

        return outputs
