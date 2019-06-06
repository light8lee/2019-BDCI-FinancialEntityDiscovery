import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sp
import math
from .layers.gat import GATLayer
from .layers import activation as Act
from .layers.pooling import GlobalMaxPooling, GlobalAvgPooling, GlobalSumPooling

class CNN_GAT(nn.Module):
    def __init__(self, vocab_size, max_seq_len, drop_rate,
                 embedding_dim, window_size, hidden_dims:list,
                 pred_dims:list, num_heads:list, cnn_dims:list,
                 init_weight=None, activation=None, pred_act:str='ELU',
                 residual:bool=False, freeze:bool=False, , add_norm=False, **kwargs):

        super(CNN_GAT, self).__init__()
        assert window_size % 2 == 1
        assert len(num_heads) == len(hidden_dims)
        assert len(cnn_dims) >= 1
        pred_act = getattr(Act, pred_act, nn.ELU)

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.drop_rate = drop_rate
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.hidden_dims = hidden_dims
        self.pred_dims = pred_dims
        self.cnn_dims = cnn_dims
        self.num_heads = num_heads
        self.freeze = freeze
        self.activation = activation
        self.add_norm = add_norm

        self.embedding = self.init_unit_embedding(init_weight=init_weight)
        self.cnn_layers = nn.ModuleList()
        self.gat_layers = nn.ModuleList()
        self.max_pool = GlobalMaxPooling()

        in_dim = self.embedding_dim
        for cnn_dim in self.cnn_dims:
            self.cnn_layers.append(
                nn.Conv1d(in_dim, cnn_dim,
                          kernel_size=window_size, stride=1, padding=window_size//2)
            )
            in_dim = cnn_dim

        if self.add_norm:
            self.norm = nn.LayerNorm(in_dim)
            self.res_weight = nn.Linear(self.embedding_dim, in_dim)

        out_dim = 0
        for hidden_dim, num_head in zip(self.hidden_dims, self.num_heads):
            self.gat_layers.append(
                GATLayer(in_dim, hidden_dim, num_head, activation, residual)
            )
            out_dim += hidden_dim
            in_dim = hidden_dim

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
        outputs = inputs.transpose(-1, -2)  # [2b, e, t]
        for conv1d in self.cnn_layers:
            outputs = F.dropout(outputs, p=self.drop_rate, training=self.training)
            outputs = conv1d(outputs)  # [2b, h, t]
            outputs = self.activation(outputs)
        outputs = outputs.transpose(-1, -2)  # [2b, t, h1]
        outputs = outputs * input_masks.unsqueeze(-1)  # [2b, t, h1]
        outputs = outputs.contiguous().view(-1, 2*self.max_seq_len, self.cnn_dims[-1])  # [b, 2t, h1]
        outputs = F.dropout(outputs, p=self.drop_rate, training=self.training)
        if self.add_norm:
            outputs = self.norm(outputs+self.res_weight(inputs))

        pooled_outputs = []
        for layer in self.gat_layers:
            outputs = F.dropout(outputs, p=self.drop_rate, training=self.training)
            outputs = layer(input_adjs, outputs)  # [b, 2t, h2]
            pooled_output = self.max_pool(outputs)
            pooled_outputs.append(pooled_output)
        pooled_outputs = torch.cat(pooled_outputs, -1)  # [b, num_gcn_layer*h2]

        outputs = self.dense(pooled_outputs)  # [b, 2]
        outputs = torch.log_softmax(outputs, 1)

        return outputs
