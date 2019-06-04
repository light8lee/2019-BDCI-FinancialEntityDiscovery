import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sp
import math
from .layers import activation as Act
from .layers.gat import GATLayer
from .layers.pooling import GlobalMaxPooling, GlobalAvgPooling, GlobalSumPooling

class RNN_GAT(nn.Module):
    def __init__(self, vocab_size, max_seq_len, drop_rate,
                 embedding_dim, hidden_dims, num_rnn_layer=1,
                 init_weight=None, activation=None,
                 freeze=False, **kwargs):

        super(RNN_GAT, self).__init__()
        assert hidden_dims[0] % 2 == 0
        assert num_rnn_layer > 0
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.drop_rate = drop_rate
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.num_rnn_layer = num_rnn_layer
        self.freeze = freeze

        self.embedding = self.init_unit_embedding(init_weight=init_weight)
        self.birnn = None  # to be replaced in subclass
        self.gcn_layers = nn.ModuleList()

        self.max_pool = GlobalMaxPooling()
        self.dense = None


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
        outputs, _ = self.birnn(outputs)  # [2b, t, h1]
        outputs = outputs * input_masks.unsqueeze(-1)  # [2b, t, h1]
        outputs = outputs.view(-1, 2*self.max_seq_len, self.hidden_dims[0])  # [b, 2t, h1]

        pooled_outputs = []
        for layer in self.gcn_layers:
            outputs = F.dropout(outputs, p=self.drop_rate, training=self.training)
            outputs = layer(input_laps, outputs)  # [b, 2t, h2]
            pooled_output = self.max_pool(outputs)  # [b, h2]
            pooled_outputs.append(pooled_output)
        pooled_outputs = torch.cat(pooled_outputs, -1)  # [b, num_layers*h2]

        outputs = self.dense(pooled_outputs)  # [b, 2]
        outputs = torch.log_softmax(outputs, 1)

        return outputs


class GRU_GAT(RNN_GAT):
    def __init__(self, vocab_size, max_seq_len, drop_rate,
                 embedding_dim, hidden_dims, num_rnn_layer=1,
                 init_weight=None, activation=None,
                 freeze:bool=False, num_heads:list=None,
                 residual:bool=False, pred_dims:list=None,
                 pred_act:str='ELU', **kwargs):
        super(GRU_GAT, self).__init__(vocab_size, max_seq_len, drop_rate,
                                      embedding_dim, hidden_dims, num_rnn_layer,
                                      init_weight, activation,
                                      freeze, **kwargs)
        num_heads = num_heads if num_heads else [1]
        pred_dims = pred_dims if pred_dims else []
        pred_act = getattr(Act, pred_act, nn.ELU)

        self.num_heads = num_heads
        self.pred_dims = pred_dims

        self.birnn = nn.GRU(self.embedding_dim, self.hidden_dims[0]//2, self.num_rnn_layer,
                            bidirectional=True, batch_first=True)

        in_dim = self.hidden_dims[0]
        flat_out_dim = 0
        for num_head in num_heads:
            self.gcn_layers.append(
                GATLayer(in_dim, self.hidden_dims[1],
                         num_head, activation=activation, residual=residual, last_layer=False)
            )
            in_dim = self.hidden_dims[1]
            flat_out_dim += self.hidden_dims[1]

        pred_layers = []
        out_dim = flat_out_dim
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


class LSTM_GAT(RNN_GAT):
    def __init__(self, vocab_size, max_seq_len, drop_rate,
                 embedding_dim, hidden_dims, num_rnn_layer=1,
                 init_weight=None, activation=None,
                 freeze:bool=False, num_heads:bool=None,
                 residual:bool=False, pred_dims:list=None,
                 pred_act:str='ELU', **kwargs):
        super(LSTM_GAT, self).__init__(vocab_size, max_seq_len, drop_rate,
                                      embedding_dim, hidden_dims, num_rnn_layer,
                                      init_weight, activation, freeze, **kwargs)
        num_heads = num_heads if num_heads else [1]
        pred_dims = pred_dims if pred_dims else []

        pred_act = getattr(Act, pred_act, nn.ELU)
        self.num_heads = num_heads
        self.pred_dims = pred_dims

        self.birnn = nn.LSTM(self.embedding_dim, self.hidden_dims[0]//2, self.num_rnn_layer,
                            bidirectional=True, batch_first=True)

        in_dim = self.hidden_dims[0]
        flat_out_dim = 0
        for num_head in num_heads:
            self.gcn_layers.append(
                GATLayer(in_dim, self.hidden_dims[1],
                         num_head, activation=activation, residual=residual, last_layer=False)
            )
            in_dim = self.hidden_dims[1]
            flat_out_dim += self.hidden_dims[1]

        pred_layers = []
        out_dim = flat_out_dim
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
