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

class BERT_Pretrained(nn.Module):
    def __init__(self, pretrained_model_path, max_seq_len, drop_rate, bert_dim,
                 gnn_hidden_dims, activation, residual, need_norm, gnn, sim="dot",
                 rescale:bool=False, adj_act="relu", **kwargs):
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
        self.crf = CRF(5, batch_first=True)

        self.bert4pretrain = BertModel.from_pretrained(pretrained_model_path)
        out_dim = bert_dim
        in_dim = bert_dim
        if gnn != "none": 
            in_size = self.max_seq_len * 2
            for i, gnn_hidden_dim in enumerate(gnn_hidden_dims):
                if rescale:
                    self.rescale_ws.append(nn.Parameter(torch.ones(1, 1, 1)))
                    self.rescale_bs.append(nn.Parameter(torch.zeros(1, 1, 1)))
                if gnn == "diffpool":
                    self.gnn_layers.append(
                        DiffPool(in_dim, gnn_hidden_dim, in_size, kwargs['ratio'],
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
                out_dim = gnn_hidden_dim
        self.hidden2tags = nn.Linear(out_dim, 5)

    def tag_outputs(self, input_ids, input_masks):
        outputs, _ = self.bert4pretrain(input_ids, attention_mask=input_masks)

        outputs = outputs * input_masks.unsqueeze(-1)

        gnn_outputs = []
        for i, gnn_layer in enumerate(self.gnn_layers):
            if self.sim == "dot":
                input_adjs = torch.bmm(outputs, outputs.transpose(-1, -2))  # [b, 2t, 2t]
            elif self.sim == "cos":
                input_adjs = torch.bmm(outputs, outputs.transpose(-1, -2))  # [b, 2t, 2t]
                norm = torch.norm(outputs, dim=-1) + 1e-7
                input_adjs = input_adjs / (norm.unsqueeze(-1) * norm.unsqueeze(1))
            elif self.sim == "self":
                input_adjs = torch.bmm(outputs, outputs.transpose(-1, -2))  # [b, 2t, 2t]
                input_adjs = input_adjs / math.sqrt(self.bert_dim)

            if self.rescale:
                input_adjs = input_adjs * self.rescale_ws[i] + self.rescale_bs[i]
            if self.need_norm:
                input_adjs = normalize_adjs(input_masks, input_adjs)
            input_adjs = self.adj_act(input_adjs)
            if self.gnn == 'diffpool':
                input_adjs, outputs = gnn_layer(input_adjs, outputs)  # [b, 2t, e]
            else:
                outputs = gnn_layer(input_adjs, outputs)
                outputs = outputs * input_masks.unsqueeze(-1)
        emissions = self.hidden2tags(outputs)
        return emissions

    def forward(self, input_ids, input_masks, target_tags):
        emissions = self.tag_outputs(input_ids, input_masks)
        scores = self.crf(emissions, target_tags, input_masks.byte())
        predicts = self.decode(emissions, input_masks)
        return scores, predicts

    def decode(self, emissions, input_masks):
        return self.crf.decode(emissions, input_masks.byte())

    def predict(self, input_ids, input_masks):
        emissions = self.tag_outputs(input_ids, input_masks)
        return self.decode(emissions, input_masks)
