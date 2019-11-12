import math
import sys
import torch
import torch.nn as nn
from .layers.mlm_bert import BertForMaskedLM_V2


POS_FLAGS = ['[PAD]', '[CLS]', '[SEP]',
             'ag', 'a', 'ad', 'an', 'b', 'c', 'dg',
             'd', 'e', 'eng', 'f', 'g', 'h', 'i', 'j', 'k',
             'l', 'm', 'ng', 'n', 'nr', 'ns', 'nt', 'nz',
             'o', 'p', 'q', 'r', 's', 'tg', 't', 'u', 'un',
             'vg', 'v', 'vd', 'vn', 'w', 'x', 'y', 'z']

class MRCBERT_Pretrained(nn.Module):
    def __init__(self, pretrained_model_path, max_seq_len, drop_rate, bert_dim,
                 rescale:bool=False, need_flags:bool=False, 
                 need_bounds:bool=False, need_birnn:bool=False, rnn:str="LSTM", rnn_dim:int=0,
                 need_extra:bool=False, num_extra:int=0,
                 lm_task:bool=False, word_seg_task:bool=False, **kwargs):
        super(MRCBERT_Pretrained, self).__init__()
        self.max_seq_len = max_seq_len
        self.drop_rate = drop_rate
        self.bert_dim = bert_dim
        self.need_flags = need_flags
        self.need_bounds = need_bounds  # do not use when do word segmentation task
        self.need_birnn = need_birnn
        self.need_extra = need_extra
        self.lm_task = lm_task

        self.bert4pretrain = BertForMaskedLM_V2.from_pretrained(pretrained_model_path)

        out_dim = bert_dim

        if self.need_birnn:
            if rnn == "LSTM":
                self.birnn = nn.LSTM(out_dim, rnn_dim, 1, bidirectional=True, batch_first=True)
            else:
                self.birnn = nn.GRU(out_dim, rnn_dim, 1, bidirectional=True, batch_first=True)
            out_dim = rnn_dim * 2
        else:
            out_dim = bert_dim

        if need_flags:
            out_dim += len(POS_FLAGS)
        if not word_seg_task and need_bounds:
            out_dim += 6
        if need_extra:
            out_dim += num_extra

        self.begin_cls = nn.Linear(out_dim, 2)
        self.end_cls = nn.Linear(out_dim, 2)
        self.span_row_fc = nn.Linear(out_dim, 1)
        self.span_col_fc = nn.Linear(out_dim, 1)
        self.drop = nn.Dropout(p=drop_rate)

    def tag_outputs(self, input_ids, input_masks,
                    flags=None, bounds=None,
                    extra=None, lm_ids=None):
        if self.lm_task:
            outputs = self.bert4pretrain(input_ids, attention_mask=input_masks,
                                                masked_lm_labels=lm_ids)
        else:
            outputs = self.bert4pretrain(input_ids, attention_mask=input_masks)
        extra_loss = outputs[0]
        seq_outputs = outputs[1]

        seq_outputs = seq_outputs * input_masks.unsqueeze(-1)

        if self.need_birnn:
            seq_outputs, *_ = self.birnn(seq_outputs)

        if self.need_flags:
            # print('outputs:', outputs.shape)
            # print('flags:', flags.shape)
            seq_outputs = torch.cat([seq_outputs, flags], -1)
        if self.need_bounds:
            seq_outputs = torch.cat([seq_outputs, bounds], -1)
        if self.need_extra:
            # print('shape:', extra.shape, file=sys.stderr)
            # print('output shape:', outputs.shape, file=sys.stderr)
            seq_outputs = torch.cat([seq_outputs, extra], -1)
        seq_outputs = self.drop(seq_outputs)  # [b, t, h]
        # if self.need_norm:
        #     seq_outputs = self.norm(seq_outputs)
        begin_emissions = self.begin_cls(seq_outputs)  # [b, t, 2]
        end_emissions = self.end_cls(seq_outputs)  # [b, t, 2]
        row_emissions = self.span_row_fc(seq_outputs)  # [b, t, 1]
        col_emissions = self.span_col_fc(seq_outputs).transpose(-1, -2)  # [b, t, 1] -> [b, 1, t]
        span_emissions = torch.sigmoid(row_emissions + col_emissions)  # [b, t, t]
        return begin_emissions, end_emissions, span_emissions, extra_loss

    def forward(self, input_ids, input_masks, target_begin_tag_ids, target_end_tag_ids, target_span_ids,
                flags=None, bounds=None, extra=None, lm_ids=None):
        begin_emissions, end_emissions, span_emissions, loss = self.tag_outputs(input_ids, input_masks,
                                     flags=flags, bounds=bounds,
                                     extra=extra, lm_ids=lm_ids)
        point_loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        # print(f"begin:{begin_emissions.shape}", file=sys.stderr)
        # print(f"target:{target_begin_tag_ids.shape}", file=sys.stderr)
        loss = loss + point_loss_fct(begin_emissions.view(-1, 2), target_begin_tag_ids.view(-1))
        loss = loss + point_loss_fct(end_emissions.view(-1, 2), target_end_tag_ids.view(-1))
        span_loss_fct = nn.BCELoss()
        loss = loss + span_loss_fct(span_emissions, target_span_ids)
        return loss

    def decode(self, emissions, input_masks):
        begin_emissions, end_emissions, span_emissions = emissions
        possible_begins = torch.argmax(begin_emissions.detach(), dim=-1).cpu().numpy()  # [b, t]
        possible_ends = torch.argmax(end_emissions.detach(), dim=-1).cpu().numpy()  # [b, t]
        span_emissions = span_emissions.cpu().numpy()  # [b, t, t]
        batch_entities = []
        for begins, ends, span in zip(possible_begins, possible_ends, span_emissions):
            entities = []
            for i, begin_score in enumerate(begins):
                if begin_score == 0:
                    continue
                for j in range(i+1, len(ends)):
                    if ends[j] == 0:
                        continue
                    if span[i][j] > 0.5:
                        entities.append((i, j))
            batch_entities.append(entities)
        return batch_entities

    def predict(self, input_ids, input_masks, flags=None, bounds=None, extra=None):
        *emissions, _ = self.tag_outputs(input_ids, input_masks, flags=flags, bounds=bounds, extra=extra)
        return self.decode(emissions, input_masks)

