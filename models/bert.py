import math
import torch
import torch.nn as nn
from .layers.diffpool import DiffPool
from .layers.graph_sage import SAGELayer
from .layers.gat import GATLayer
from .layers.gcn import GCNLayer
from .layers import activation as Act
from .layers.pooling import MaxPooling, AvgPooling, SumPooling
from .layers.normalization import normalize_adjs
from torchcrf import CRF
from pytorch_transformers import BertModel

POS_FLAGS = ['[PAD]', '[CLS]', '[SEP]', 
             'ag', 'a', 'ad', 'an', 'b', 'c', 'dg',
             'd', 'e', 'eng', 'f', 'g', 'h', 'i', 'j', 'k',
             'l', 'm', 'ng', 'n', 'nr', 'ns', 'nt', 'nz', 
             'o', 'p', 'q', 'r', 's', 'tg', 't', 'u', 'un', 
             'vg', 'v', 'vd', 'vn', 'w', 'x', 'y', 'z']

class BERT_Pretrained(nn.Module):
    def __init__(self, pretrained_model_path, max_seq_len, drop_rate, bert_dim,
                 rescale:bool=False, need_flags:bool=False, adj_act="relu", need_bounds:bool=False,
                 need_birnn:bool=False, rnn="LSTM", rnn_dim=0, **kwargs):
        super(BERT_Pretrained, self).__init__()
        self.max_seq_len = max_seq_len
        self.drop_rate = drop_rate
        self.gnn_layers = nn.ModuleList()  # compatable needs
        self.bert_dim = bert_dim
        self.rescale_ws = nn.ParameterList()  # compatable needs
        self.rescale_bs = nn.ParameterList()  # compatable needs
        self.need_flags = need_flags
        self.need_bounds = need_bounds
        self.need_birnn = need_birnn
        self.crf = CRF(5, batch_first=True)

        self.bert4pretrain = BertModel.from_pretrained(pretrained_model_path)
        if self.need_birnn:
            if rnn == "LSTM":
                self.birnn = nn.LSTM(bert_dim, rnn_dim, 1, bidirectional=True, batch_first=True)
            else:
                self.birnn = nn.GRU(bert_dim, rnn_dim, 1, bidirectional=True, batch_first=True)
            out_dim = rnn_dim * 2
        else:
            out_dim = bert_dim

        if need_flags:
            out_dim += len(POS_FLAGS)
        if need_bounds:
            out_dim += 6
        
        self.hidden2tags = nn.Linear(out_dim, 5)

    def tag_outputs(self, input_ids, input_masks, flags=None, bounds=None):
        outputs, _ = self.bert4pretrain(input_ids, attention_mask=input_masks)

        outputs = outputs * input_masks.unsqueeze(-1)

        if self.need_birnn:
            outputs, *_ = self.birnn(outputs)

        if self.need_flags:
            # print('outputs:', outputs.shape)
            # print('flags:', flags.shape)
            outputs = torch.cat([outputs, flags], -1)
        if self.need_bounds:
            outputs = torch.cat([outputs, bounds], -1)
        emissions = self.hidden2tags(outputs)
        return emissions

    def forward(self, input_ids, input_masks, target_tags, flags=Non, bounds=Nonee):
        emissions = self.tag_outputs(input_ids, input_masks, flags=flags, bounds=bounds)
        scores = self.crf(emissions, target_tags, input_masks.byte())
        return scores

    def decode(self, emissions, input_masks):
        return self.crf.decode(emissions, input_masks.byte())

    def predict(self, input_ids, input_masks, flags=None, bounds=None):
        emissions = self.tag_outputs(input_ids, input_masks, flags=flags, bounds=bounds)
        return self.decode(emissions, input_masks)
