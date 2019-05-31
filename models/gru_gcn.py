import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sp
import math
from .layers.gcn import GCNLayer
from .layers.pooling import GlobalMaxPooling, GlobalAvgPooling, GlobalSumPooling

class GRU_GCN(nn.Module):
    def __init__(self, vocab_size, max_seq_len, drop_rate,
                 embedding_dim, hidden_dims, num_gru_layer=1,
                 num_gcn_layer=1, init_weight=None, activation=None, **kwargs):

        super(GRU_GCN, self).__init__()
        assert hidden_dims[0] % 2 == 0
        assert num_gcn_layer > 0
        assert num_gru_layer > 0
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.drop_rate = drop_rate
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.num_gru_layer = num_gru_layer
        self.num_gcn_layer = num_gcn_layer

        self.embedding = self.init_unit_embedding(init_weight=init_weight)
        self.bigru = nn.GRU(embedding_dim, hidden_dims[0]//2, self.num_gru_layer,
                            bidirectional=True, batch_first=True)
        self.gcn_layers = [GCNLayer(hidden_dims[0], hidden_dims[1], activation)]
        for _ in range(num_gcn_layer-1):
            self.gcn_layers.append(
                GCNLayer(hidden_dims[1], hidden_dims[1], activation)
            )
        self.gcn_layers = nn.ModuleList(self.gcn_layers)
        self.mean_pool = GlobalAvgPooling()
        self.max_pool = GlobalMaxPooling()
        self.sum_pool = GlobalSumPooling()

        self.dense = nn.Linear(2*hidden_dims[1], 2)

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
        outputs, hn = self.bigru(outputs)  # [2b, t, h1]
        outputs = outputs * input_masks.unsqueeze(-1)  # [2b, t, h1]
        outputs = outputs.view(-1, 2*self.max_seq_len, self.hidden_dims[0])  # [b, 2t, h1]
        pooled_outputs = []
        for layer in self.gcn_layers:
            outputs = F.dropout(outputs, p=self.drop_rate, training=self.training)
            outputs = layer(input_laps, outputs)  # [b, 2t, h2]
            pooled_output = [self.max_pool(outputs), self.mean_pool(outputs)]
            pooled_output = torch.cat(pooled_output, 1)  # [b, 2h2]
            pooled_outputs.append(pooled_output)
        pooled_outputs = torch.stack(pooled_outputs)  # [num_gcn_layer, b, 2h2]
        pooled_outputs = self.sum_pool(pooled_outputs)  # [b, 2h2]
        num_nodes = torch.sum(input_masks.view(-1, 2*self.max_seq_len), 1)  # [2b, t] -> [b, 2t] -> [b]
        pooled_outputs = pooled_outputs / num_nodes.unsqueeze(-1)

        outputs = self.dense(pooled_outputs)  # [b, 2]
        outputs = torch.log_softmax(outputs, 1)

        return outputs

