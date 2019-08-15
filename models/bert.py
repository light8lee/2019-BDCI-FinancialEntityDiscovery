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
    def __init__(self, pretrained_model_path, max_seq_len, drop_rate, readout_pool, bert_dim,
                 gnn_hidden_dims, activation, residual, need_norm, gnn, sim="dot",
                 adj_act="relu", pred_dims=None, pred_act='ELU', **kwargs):
        super(BERT_Pretrained, self).__init__()
        assert sim in ["dot", "cos"]
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

        if readout_pool == 'max':
            self.readout_pool = MaxPooling()
        elif readout_pool == 'avg':
            self.readout_pool = AvgPooling()
        elif readout_pool == 'sum':
            self.readout_pool = SumPooling()
        else:
            raise ValueError()

        self.bert4pretrain = BertForPreTraining.from_pretrained(pretrained_model_path, from_tf=True).bert
        out_dim = bert_dim * 4
        if gnn != "none": 
            in_size = self.max_seq_len * 2
            for i, gnn_hidden_dim in enumerate(gnn_hidden_dims):
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
                out_dim += in_dim

        pred_act = getattr(Act, pred_act, nn.ELU)

        self.pred_dims = pred_dims
        pred_layers = []
        for pred_dim in pred_dims:
            pred_layers.append(
                nn.Linear(out_dim, pred_dim)
            )
            pred_layers.append(pred_act())
            pred_layers.append(nn.Dropout(p=self.drop_rate))
            out_dim = pred_dim
        pred_layers.append(
            nn.Linear(out_dim, 1)
        )

        self.dense = nn.Sequential(*pred_layers)

    def forward(self, input_ids, input_masks):
        """ 由training来控制finetune还是固定 """

        inputs_a, inputs_b = input_ids
        masks_a, masks_b = input_masks
        self.bert4pretrain.eval()
        with t.no_grad():
            outputs_a, _ = self.bert4pretrain(inputs_a, attention_mask=masks_a, output_all_encoded_layers=False)
            outputs_b, _ = self.bert4pretrain(inputs_b, attention_mask=masks_b, output_all_encoded_layers=False)

        sim_outputs = []
        pool_a = self.readout_pool(outputs_a, 1)
        pool_b = self.readout_pool(outputs_b, 1)
        # sim_outputs.append(self._cos_sim(pool_a, pool_b).unsqueeze(-1))
        sim_outputs.append(pool_a)
        sim_outputs.append(pool_b)
        sim_outputs.append(torch.abs(pool_a - pool_b))
        sim_outputs.append(pool_a * pool_b)

        outputs = torch.cat([outputs_a, outputs_b], 1)  # [b, 2t, e]
        for gnn_layer in self.gnn_layers:
            if self.sim == "dot":
                input_adjs = torch.bmm(outputs, outputs.transpose(-1, -2))  # [b, 2t, 2t]
            elif self.sim == "cos":
                input_adjs = torch.bmm(outputs, outputs.transpose(-1, -2))  # [b, 2t, 2t]
                norm = torch.norm(outputs, dim=-1) + 1e-7
                input_adjs = input_adjs / (norm.unsqueeze(-1) * norm.unsqueeze(1))
            if self.need_norm:
                input_adjs = normalize_adjs(input_masks, input_adjs)
            input_adjs = self.adj_act(input_adjs)
            if self.gnn == 'diffpool':
                input_adjs, outputs = gnn_layer(input_adjs, outputs)  # [b, 2t, e]
                sim_outputs.append(self.readout_pool(outputs, 1))
            else:
                outputs = gnn_layer(input_adjs, outputs)
                sim_outputs.append(self.readout_pool(outputs, 1))

        outputs = torch.cat(sim_outputs, -1)
        outputs = self.dense(outputs)  # [b, 1]
        # outputs = torch.log_softmax(outputs, 1)

        return outputs