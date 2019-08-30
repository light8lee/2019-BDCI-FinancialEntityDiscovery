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
from pytorch_pretrained_bert import BertModel, BertConfig, BertForPreTraining

class BERT_Pretrained(nn.Module):
    def __init__(self, output_dim, pretrained_model_path, max_seq_len, drop_rate, readout_pool, bert_dim,
                 gnn_hidden_dims, activation, residual, need_norm, gnn, sim="dot",
                 rescale:bool=False, adj_act="relu", pred_dims=None, pred_act='ELU', **kwargs):
        super(BERT_Pretrained, self).__init__()
        assert sim in ["dot", "cos", "self"]
        assert gnn in ["diffpool", "gcn", "gat", "none"]
        pred_dims = pred_dims if pred_dims else []
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

        if readout_pool == 'max':
            self.readout_pool = MaxPooling()
        elif readout_pool == 'avg':
            self.readout_pool = AvgPooling()
        elif readout_pool == 'sum':
            self.readout_pool = SumPooling()
        else:
            raise ValueError()

        self.bert4pretrain = BertForPreTraining.from_pretrained(pretrained_model_path, from_tf=True).bert
        out_dim = pred_dims[0]
        self.pooled_fc = nn.Linear(bert_dim, pred_dims[0])
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
            out_dim *= 2
            self.gnn_fc = nn.Linear(sum(gnn_hidden_dims), pred_dims[0])
        else:
            self.gnn_fc = None

        pred_act = getattr(Act, pred_act, nn.ELU)

        self.pred_dims = pred_dims
        pred_layers = []
        for pred_dim in pred_dims[1:]:
            pred_layers.append(
                nn.Linear(out_dim, pred_dim)
            )
            pred_layers.append(pred_act())
            pred_layers.append(nn.Dropout(p=self.drop_rate))
            out_dim = pred_dim
        pred_layers.append(
            nn.Linear(out_dim, output_dim)
        )

        self.dense = nn.Sequential(*pred_layers)

    def forward(self, input_ids, input_masks, input_types):
        outputs, pooled_outputs = self.bert4pretrain(input_ids, token_type_ids=input_types, attention_mask=input_masks, output_all_encoded_layers=False)

        sim_outputs = []
        outputs = outputs * input_masks.unsqueeze(-1)
        sim_outputs.append(
            self.pooled_fc(pooled_outputs)
        )
        
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
            gnn_outputs.append(self.readout_pool(outputs, 1))
        if self.gnn_fc:
            gnn_outputs = torch.cat(gnn_outputs, -1)
            sim_outputs.append(
                self.gnn_fc(gnn_outputs)
            )

        outputs = torch.cat(sim_outputs, -1)
        outputs = self.dense(outputs)  # [b, 1]

        return outputs