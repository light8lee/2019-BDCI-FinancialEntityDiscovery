import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sp
import math
from itertools import zip_longest
from .layers.gat import GATLayer
from .layers.gcn import GCNLayer
from .layers.graph_sage import SAGELayer
from .layers.diffpool import DiffPool
from .layers.abcnn import ABCNN1
from .layers import activation as Act
from .layers.pooling import MaxPooling, AvgPooling, SumPooling

class GAT_ABCNN1(nn.Module):
    def __init__(self, vocab_size, max_seq_len, drop_rate, gnn_hidden_dims:list,
                 embedding_dim, window_size, cnn_hidden_dims:list, attn,
                 readout_pool:str, mode:str='concat', pred_dims:list=None,
                 need_norm:bool=False, gnn_channels:list=None,
                 init_weight=None, activation=None, pred_act:str='ELU', need_embed:bool=False,
                 residual:bool=False, freeze:bool=False, **kwargs):

        super(GAT_ABCNN1, self).__init__()
        assert window_size % 2 == 1
        assert mode in ['cos', 'concat']
        gnn_channels = ["gcn", "gat"] if gnn_channels is None else gnn_channels
        pred_act = getattr(Act, pred_act, nn.ELU)

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.drop_rate = drop_rate
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.cnn_hidden_dims = cnn_hidden_dims
        self.gnn_hidden_dims = gnn_hidden_dims
        self.pred_dims = pred_dims
        # self.num_heads = num_heads
        self.gnn_channels = gnn_channels
        self.freeze = freeze
        self.activation = getattr(Act, activation)
        self.mode = mode
        self.need_embed = need_embed
        self.need_norm = need_norm
        self.extra_dim = extra_dim

        self.embedding = self.init_unit_embedding(init_weight=init_weight)
        self.cnn_layers = nn.ModuleList()
        self.outer_gat_layers = nn.ModuleList()
        self.outer_gcn_layers = nn.ModuleList()
        self.outer_sage_layers = nn.ModuleList()
        self.outer_diffpool_layers = nn.ModuleList()

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
            assert len(num_heads) == len(gnn_hidden_dims)
        else:
            num_heads = []
        for i, hidden_dim in enumerate(cnn_hidden_dims):
            # if "gat" in gnn_channels:
            #     self.outer_gat_layers.append(
            #         GATLayer(in_dim, in_dim, num_heads[i], self.activation, residual=residual, last_layer=False)
            #     )
            # if "gcn" in gnn_channels:
            #     self.outer_gcn_layers.append(
            #         GCNLayer(in_dim, in_dim, self.activation, residual=residual)
            #     )
            self.cnn_layers.append(
                ABCNN1(in_dim, hidden_dim, max_seq_len, window_size, self.activation,
                       num_extra_channel=0, attn=attn)
            )
            in_dim = hidden_dim
        
        out_dim = in_dim * 4

        if "diffpool" in gnn_channels:
            in_size = max_seq_len * 2
            for gnn_hidden_dim in zip(gnn_hidden_dims):
                self.outer_diffpool_layers.append(
                    DiffPool(in_dim, in_size, kwargs['ratio'],
                             gnn='gcn', activation=self.activation, residual=residual)
                )
                in_size = int(in_size * kwargs['ratio'])
                out_dim += in_dim
        else:
            for num_head, gnn_hidden_dim in zip_longest(num_heads, gnn_hidden_dims):
                if "gat" in gnn_channels:
                    self.outer_gat_layers.append(
                        GATLayer(in_dim, gnn_hidden_dim, num_head, self.activation, residual=residual, last_layer=False)
                    )
                    out_dim += gnn_hidden_dim
                if "gcn" in gnn_channels:
                    self.outer_gcn_layers.append(
                        GCNLayer(in_dim, gnn_hidden_dim, self.activation, residual=residual)
                    )
                    out_dim += gnn_hidden_dim
                if "sage" in gnn_channels:
                    self.outer_sage_layers.append(
                        SAGELayer(in_dim, gnn_hidden_dim, self.activation, kwargs['pooling'])
                    )
                    out_dim += gnn_hidden_dim

        if mode == 'concat':
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
        elif mode == 'cos':
            self.dense = nn.Sequential(
                nn.Linear(4, 2)
            )

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
            norm_adjs = input_adjs * normalization
            input_adjs = norm_adjs.masked_fill(input_adjs==0, 0)
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
        len_a = torch.sum(masks_a, -1) / 80
        len_b = torch.sum(masks_b, -1) / 80
        input_masks = torch.cat(input_masks, 1)  # [b, 2t]
        
        inputs_a = self.embedding(inputs_a)  # [b, t, e]
        inputs_b = self.embedding(inputs_b)  # [b, t, e]

        input_adjs = self._normalize_adjs(input_masks, input_adjs)

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

        outputs_a = inputs_a
        outputs_b = inputs_b

        for i, cnn_layer in enumerate(self.cnn_layers):
            outputs = torch.cat([outputs_a, outputs_b], 1)  # [b, 2t, e]

            outputs_a, outputs_b = cnn_layer(outputs_a, outputs_b)
            outputs_a = outputs_a * masks_a
            outputs_b = outputs_b * masks_b

        if self.mode == 'concat':
            sim_outputs = []

            pool_a = self.readout_pool(outputs_a, 1)
            pool_b = self.readout_pool(outputs_b, 1)
            # sim_outputs.append(self._cos_sim(pool_a, pool_b).unsqueeze(-1))
            sim_outputs.append(pool_a)
            sim_outputs.append(pool_b)
            sim_outputs.append(torch.abs(pool_a - pool_b))
            sim_outputs.append(pool_a * pool_b)

            if 'diffpool' in self.gnn_channels:
                outputs = torch.cat([outputs_a, outputs_b], 1)  # [b, 2t, e]
                for diffpool_layer in self.outer_diffpool_layers:
                    input_adjs, outputs = diffpool_layer(input_adjs, outputs)
                    sim_outputs.append(
                        self.readout_pool(outputs, 1)
                    )
            else:
                sage_outputs = gcn_outputs = gat_outputs = torch.cat([outputs_a, outputs_b], 1)  # [b, 2t, e]
                for gat_layer, gcn_layer, sage_layer in zip_longest(self.outer_gat_layers, self.outer_gcn_layers, self.outer_sage_layers):
                    if gat_layer:
                        gat_outputs = gat_layer(input_adjs, gat_outputs)  # [b, 2t, e]
                        sim_outputs.append(self.readout_pool(gat_outputs, 1))

                    if gcn_layer:
                        gcn_outputs = gcn_layer(input_adjs, gcn_outputs)
                        sim_outputs.append(self.readout_pool(gcn_outputs, 1))

                    if sage_layer:
                        sage_outputs = sage_layer(input_adjs, sage_outputs)
                        sim_outputs.append(self.readout_pool(sage_outputs, 1))
            outputs = torch.cat(sim_outputs, -1)  # [b, h]
            # outputs = self.concat_norm(outputs)
        elif self.mode == 'cos':
            sim_outputs = [len_a, len_b]
            pool_a = self.readout_pool(inputs_a, 1)
            pool_b = self.readout_pool(inputs_b, 1)
            sim_outputs.append(
                self._cos_sim(pool_a, pool_b)
            )

            pool_a = self.readout_pool(outputs_a, 1)
            pool_b = self.readout_pool(outputs_b, 1)
            sim_outputs.append(
                self._cos_sim(pool_a, pool_b)
            )
            outputs = torch.stack(sim_outputs, -1)  # [b, 2]

        outputs = self.dense(outputs)
        outputs = torch.log_softmax(outputs, 1)

        return outputs