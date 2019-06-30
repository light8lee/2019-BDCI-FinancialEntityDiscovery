import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sp
import math
from .layers.gat import GATLayer
from .layers.gcn import GCNLayer
from .layers.abcnn import ABCNN1
from .layers import activation as Act
from .layers.pooling import MaxPooling, AvgPooling, SumPooling

class GAT_ABCNN1(nn.Module):
    def __init__(self, vocab_size, max_seq_len, drop_rate,
                 embedding_dim, window_size, hidden_dims:list,
                 pred_dims:list, readout_pool:str, need_norm:bool=False, gnn_channels:list=None,
                 init_weight=None, activation=None, pred_act:str='ELU', need_embed:bool=False,
                 residual:bool=False, freeze:bool=False, mode='add_norm', **kwargs):

        super(GAT_ABCNN1, self).__init__()
        assert window_size % 2 == 1
        gnn_channels = ["gcn", "gat"] if gnn_channels is None else gnn_channels
        pred_act = getattr(Act, pred_act, nn.ELU)

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.drop_rate = drop_rate
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.hidden_dims = hidden_dims
        self.pred_dims = pred_dims
        # self.num_heads = num_heads
        self.gnn_channels = gnn_channels
        self.freeze = freeze
        self.activation = activation
        self.mode = mode
        self.need_embed = need_embed
        self.need_norm = need_norm

        self.embedding = self.init_unit_embedding(init_weight=init_weight)
        self.cnn_layers = nn.ModuleList()
        # self.inter_gat_layers = nn.ModuleList()
        self.outer_gat_layers = nn.ModuleList()
        self.outer_gcn_layers = nn.ModuleList()

        if readout_pool == 'max':
            self.readout_pool = MaxPooling()
        elif readout_pool == 'avg':
            self.readout_pool = AvgPooling()
        elif readout_pool == 'sum':
            self.readout_pool = SumPooling()
        else:
            raise ValueError()

        in_dim = self.embedding_dim
        if "gat" in gnn_channels:
            num_heads = kwargs['num_heads']
            assert len(num_heads) == len(hidden_dims)
        for i, hidden_dim in enumerate(hidden_dims):
            # self.inter_gat_layers.append(
            #     GATLayer(in_dim, hidden_dim, num_head, activation, residual=residual, last_layer=False)
            # )
            if "gat" in gnn_channels:
                self.outer_gat_layers.append(
                    GATLayer(in_dim, in_dim, num_heads[i], activation, residual=residual, last_layer=False)
                )
            if "gcn" in gnn_channels:
                self.outer_gcn_layers.append(
                    GCNLayer(in_dim, in_dim, activation, residual=residual)
                )
            self.cnn_layers.append(
                ABCNN1(in_dim, hidden_dim, max_seq_len, window_size, activation, num_channel=2+len(gnn_channels))
            )
            in_dim = hidden_dim
        out_dim = sum(hidden_dims) * 4 # + 4
        self.concat_norm = nn.LayerNorm(out_dim)
        pred_layers = []
        for pred_dim in pred_dims:
            pred_layers.append(
                nn.Linear(out_dim, pred_dim)
            )
            pred_layers.append(pred_act())
            pred_layers.append(nn.Dropout(p=self.drop_rate))
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
    
    def _normalize_adjs(self, input_masks, input_adjs):
        selfloop = torch.eye(self.max_seq_len*2, device=input_adjs.device).unsqueeze(0)  # [1, t, t]
        selfloop = selfloop * input_masks.unsqueeze(-1)  # [b, t, t], keep padding values to 0
        input_adjs = input_adjs + selfloop  # [b, t, t]
        if self.need_norm:  # this is equvalient to D^{-1/2} A D^{-1/2}
            row_sum = input_adjs.sum(-1)  # [b, t]
            d_inv_sqrt = torch.pow(row_sum, -0.5)  # [b, t]
            d_inv_sqrt *= input_masks  # [b, t], keep padding values to 0
            normalization = d_inv_sqrt.unsqueeze(-1) * d_inv_sqrt.unsqueeze(1) # [b, t, t]
            input_adjs *= normalization
        return input_adjs

    def _cos_sim(self, inputs_a, inputs_b):
        norm_a = torch.norm(inputs_a)  # [b]
        norm_b = torch.norm(inputs_a)  # [b]
        product = torch.sum(inputs_a * inputs_b, 1)  # [b]
        return product / (norm_a * norm_b)

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
        input_masks = torch.cat(input_masks, 1)  # [b, 2t]
        
        inputs_a = self.embedding(inputs_a)  # [b, t, e]
        inputs_b = self.embedding(inputs_b)  # [b, t, e]

        input_adjs = [
            self._normalize_adjs(input_masks, input_adjs[0]),
            self._normalize_adjs(input_masks, input_adjs[1]),
        ]

        sim_outputs = []
        # sim_outputs.append(
        #     self._cos_sim(
        #         self.readout_pool(inputs_a, 1),
        #         self.readout_pool(inputs_b, 1)
        #     ).unsqueeze(-1)
        # )

        # pool_a = self.readout_pool(inputs_a, -1)
        # pool_b = self.readout_pool(inputs_b, -1)
        # sim_outputs.append(self._cos_sim(pool_a, pool_b))
        masks_a = masks_a.unsqueeze(-1)
        masks_b = masks_b.unsqueeze(-1)

        # for outer_gcn_layer, outer_gat_layer, cnn_layer in zip(self.outer_gcn_layers, self.outer_gat_layers, self.cnn_layers):
        for i, cnn_layer in enumerate(self.cnn_layers):
            outputs = torch.cat([inputs_a, inputs_b], 1)  # [b, 2t, e]

            extra_a_inputs = []
            extra_b_inputs = []

            # gat_outputs = inter_gat_layer(input_adjs[0], outputs)  # [b, 2t, e]
            # gat_a_outputs, gat_b_outputs = torch.chunk(gat_outputs, 2, 1)  # [b, t, e] * 2

            # extra_a_inputs.append(gat_a_outputs * masks_a)
            # extra_b_inputs.append(gat_b_outputs * masks_b)

            if "gat" in self.gnn_channels:
                gat_outputs = self.outer_gat_layers[i](input_adjs[1], outputs)  # [b, 2t, e]
                gat_a_outputs, gat_b_outputs = torch.chunk(gat_outputs, 2, 1)  # [b, t, e] * 2
                # sim_outputs.append(
                #     self._cos_sim(
                #         self.readout_pool(gat_a_outputs, 1),
                #         self.readout_pool(gat_b_outputs, 1)
                #     ).unsqueeze(-1)
                # )

                extra_a_inputs.append(gat_a_outputs * masks_a)
                extra_b_inputs.append(gat_b_outputs * masks_b)

            if "gcn" in self.gnn_channels:
                gcn_outputs = self.outer_gcn_layers[i](input_adjs[1], outputs)
                gcn_a_outputs, gcn_b_outputs = torch.chunk(gcn_outputs, 2, 1)  # [b, t, e] * 2
                # sim_outputs.append(
                #     self._cos_sim(
                #         self.readout_pool(gcn_a_outputs, 1),
                #         self.readout_pool(gcn_b_outputs, 1)
                #     ).unsqueeze(-1)
                # )

                extra_a_inputs.append(gcn_a_outputs * masks_a)
                extra_b_inputs.append(gcn_b_outputs * masks_b)

            inputs_a, inputs_b = cnn_layer(inputs_a, inputs_b, extra_a_inputs, extra_b_inputs)
            inputs_a = inputs_a * masks_a
            inputs_b = inputs_b * masks_b

            pool_a = self.readout_pool(inputs_a, 1)
            pool_b = self.readout_pool(inputs_b, 1)
            # sim_outputs.append(self._cos_sim(pool_a, pool_b).unsqueeze(-1))
            sim_outputs.append(pool_a)
            sim_outputs.append(pool_b)
            sim_outputs.append(torch.abs(pool_a - pool_b))
            sim_outputs.append(pool_a * pool_b)
        outputs = torch.cat(sim_outputs, -1)  # [b, h]
        # outputs = self.concat_norm(outputs)

        outputs = self.dense(outputs)
        outputs = torch.log_softmax(outputs, 1)

        return outputs