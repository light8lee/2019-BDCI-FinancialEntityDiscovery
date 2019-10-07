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
                 gnn_hidden_dims, activation, residual, need_norm, gnn, sim="dot",
                 rescale:bool=False, need_flags:bool=False, adj_act="relu", **kwargs):
        super(BERT_Pretrained, self).__init__()
        assert sim in ["dot", "cos", "self"]
        assert gnn in ["diffpool", "gcn", "gat", "none"]
        self.max_seq_len = max_seq_len
        self.drop_rate = drop_rate
        self.gnn_hidden_dims = gnn_hidden_dims
        self.activation = getattr(Act, activation)
        self.need_norm = need_norm
        self.gnn = gnn
        self.sim = sim
        self.adj_act = getattr(Act, adj_act)
        self.gnn_layers = nn.ModuleList()
        self.rescale = rescale
        self.bert_dim = bert_dim
        self.rescale_ws = nn.ParameterList()
        self.rescale_bs = nn.ParameterList()
        self.need_flags = need_flags
        self.crf = CRF(5, batch_first=True)

        self.bert4pretrain = BertModel.from_pretrained(pretrained_model_path)
        out_dim = bert_dim if not need_flags else (bert_dim+len(POS_FLAGS))
        
        self.hidden2tags = nn.Linear(out_dim, 5)

    def tag_outputs(self, input_ids, input_masks, flags=None):
        outputs, _ = self.bert4pretrain(input_ids, attention_mask=input_masks)

        outputs = outputs * input_masks.unsqueeze(-1)

        if self.need_flags:
            outputs = torch.concat([outputs, flags], -1)
        emissions = self.hidden2tags(outputs)
        return emissions

    def forward(self, input_ids, input_masks, target_tags, flags=None):
        emissions = self.tag_outputs(input_ids, input_masks, flags=flags)
        scores = self.crf(emissions, target_tags, input_masks.byte())
        return scores

    def decode(self, emissions, input_masks):
        return self.crf.decode(emissions, input_masks.byte())

    def predict(self, input_ids, input_masks, flags=None):
        emissions = self.tag_outputs(input_ids, input_masks, flags=flags)
        return self.decode(emissions, input_masks)
