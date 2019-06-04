import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sp
import math
from .layers import activation as Act
from .layers.pooling import GlobalMaxPooling, GlobalAvgPooling, GlobalSumPooling

class RNN(nn.Module):
    def __init__(self, vocab_size, max_seq_len, drop_rate,
                 embedding_dim, hidden_dim, num_rnn_layer=1,
                 init_weight=None, freeze:bool=False,
                 pred_dims:list=None, pred_act:str='ELU', **kwargs):
        super(RNN, self).__init__()
        assert hidden_dim % 2 == 0
        assert num_rnn_layer > 0
        pred_dims = pred_dims if pred_dims else []

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.drop_rate = drop_rate
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_rnn_layer = num_rnn_layer
        self.freeze = freeze

        self.embedding = self.init_unit_embedding(init_weight=init_weight)
        self.birnn = None  # to be replaced in subclass

        self.max_pool = GlobalMaxPooling()
        pred_act = getattr(Act, pred_act, nn.ELU)

        self.pred_dims = pred_dims

        pred_layers = []
        out_dim = hidden_dim * 2
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
        outputs, _ = self.birnn(outputs)  # [2b, t, h]
        outputs = outputs * input_masks.unsqueeze(-1)  # [2b, t, h]

        outputs = self.max_pool(outputs)  # [2b, h]
        outputs = outputs.view(-1, 2*self.hidden_dim)  # [b, 2h]

        outputs = self.dense(outputs)  # [b, 2]
        outputs = torch.log_softmax(outputs, 1)

        return outputs


class GRU(RNN):
    def __init__(self, vocab_size, max_seq_len, drop_rate,
                 embedding_dim, hidden_dim, num_rnn_layer=1,
                 init_weight=None, freeze:bool=False,
                 pred_dims:list=None, pred_act:str='ELU', **kwargs):
        super(GRU, self).__init__(vocab_size, max_seq_len, drop_rate,
                                  embedding_dim, hidden_dim, num_rnn_layer,
                                  init_weight, freeze, pred_dims, pred_act, **kwargs)
        self.birnn = nn.GRU(self.embedding_dim, self.hidden_dim//2, self.num_rnn_layer,
                            bidirectional=True, batch_first=True)


class LSTM(RNN):
    def __init__(self, vocab_size, max_seq_len, drop_rate,
                 embedding_dim, hidden_dim, num_rnn_layer=1,
                 init_weight=None, freeze:bool=False,
                 pred_dims:list=None, pred_act:str='ELU', **kwargs):
        super(LSTM, self).__init__(vocab_size, max_seq_len, drop_rate,
                                  embedding_dim, hidden_dim, num_rnn_layer,
                                  init_weight, freeze, pred_dims, pred_act, **kwargs)
        self.birnn = nn.LSTM(self.embedding_dim, self.hidden_dim//2, self.num_rnn_layer,
                            bidirectional=True, batch_first=True)