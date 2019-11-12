import math
import sys
import torch
import torch.nn as nn
from torchcrf import CRF
from .layers import activation as Act
from .layers.mlm_bert import BertForMaskedLM_V2, BertModel


POS_FLAGS = ['[PAD]', '[CLS]', '[SEP]',
             'ag', 'a', 'ad', 'an', 'b', 'c', 'dg',
             'd', 'e', 'eng', 'f', 'g', 'h', 'i', 'j', 'k',
             'l', 'm', 'ng', 'n', 'nr', 'ns', 'nt', 'nz',
             'o', 'p', 'q', 'r', 's', 'tg', 't', 'u', 'un',
             'vg', 'v', 'vd', 'vn', 'w', 'x', 'y', 'z']

class BERT_Pretrained(nn.Module):
    def __init__(self, pretrained_model_path, max_seq_len, drop_rate, bert_dim,
                 rescale:bool=False, need_flags:bool=False, num_tag:int=5,
                 need_bounds:bool=False, need_birnn:bool=False, rnn:str="LSTM", rnn_dim:int=0,
                 need_extra:bool=False, num_extra:int=0, inner_layers:list=None,
                 lm_task:bool=False, word_seg_task:bool=False, **kwargs):
        super(BERT_Pretrained, self).__init__()
        self.max_seq_len = max_seq_len
        self.drop_rate = drop_rate
        self.bert_dim = bert_dim
        self.need_flags = need_flags
        self.need_bounds = need_bounds  # do not use when do word segmentation task
        self.need_birnn = need_birnn
        # self.need_norm = need_norm
        self.inner_layers = inner_layers
        self.need_extra = need_extra
        self.num_tag = num_tag
        self.lm_task = lm_task
        self.word_seg_task = word_seg_task
        self.crf = CRF(num_tag, batch_first=True)

        self.bert4pretrain = BertForMaskedLM_V2.from_pretrained(pretrained_model_path)

        out_dim = bert_dim
        if inner_layers is not None:
            self.bert4pretrain.bert.encoder.output_hidden_states = True
            # self.proj = nn.Linear(len(inner_layers)*bert_dim, bert_dim)
            out_dim = len(inner_layers) * bert_dim

        if self.need_birnn:
            if rnn == "LSTM":
                self.birnn = nn.LSTM(out_dim, rnn_dim, 1, bidirectional=True, batch_first=True)
            else:
                self.birnn = nn.GRU(out_dim, rnn_dim, 1, bidirectional=True, batch_first=True)
            out_dim = rnn_dim * 2
        else:
            out_dim = bert_dim

        # if need_norm:
        #     self.norm = nn.LayerNorm(out_dim)

        if need_flags:
            out_dim += len(POS_FLAGS)
        if not word_seg_task and need_bounds:
            out_dim += 6
        if need_extra:
            out_dim += num_extra

        self.hidden2tags = nn.Linear(out_dim, num_tag)
        if word_seg_task:
            self.seg_crf = CRF(6, batch_first=True)
            self.seg_hidden2tags = nn.Linear(out_dim, 6)
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
        if self.inner_layers is None:
            seq_outputs = outputs[1]
        else:
            states = outputs[2]
            seq_outputs = torch.cat([states[i] for i in self.inner_layers], dim=-1)
            # seq_outputs = self.proj(seq_outputs)

        seq_outputs = seq_outputs * input_masks.unsqueeze(-1)

        if self.need_birnn:
            seq_outputs, *_ = self.birnn(seq_outputs)

        if self.need_flags:
            # print('outputs:', outputs.shape)
            # print('flags:', flags.shape)
            seq_outputs = torch.cat([seq_outputs, flags], -1)
        if not self.word_seg_task and self.need_bounds:
            seq_outputs = torch.cat([seq_outputs, bounds], -1)
        if self.need_extra:
            # print('shape:', extra.shape, file=sys.stderr)
            # print('output shape:', outputs.shape, file=sys.stderr)
            seq_outputs = torch.cat([seq_outputs, extra], -1)
        seq_outputs = self.drop(seq_outputs)
        # if self.need_norm:
        #     seq_outputs = self.norm(seq_outputs)
        emissions = self.hidden2tags(seq_outputs)
        if self.word_seg_task:
            seg_emissions = self.seg_hidden2tags(seq_outputs)
            seg_scores = self.seg_crf(seg_emissions, torch.argmax(bounds, dim=-1), input_masks.byte())
            extra_loss = extra_loss - seg_scores
        return emissions, extra_loss

    def forward(self, input_ids, input_masks, target_tags,
                flags=None, bounds=None, extra=None, lm_ids=None):
        emissions, loss = self.tag_outputs(input_ids, input_masks,
                                     flags=flags, bounds=bounds,
                                     extra=extra, lm_ids=lm_ids)
        loss = loss - self.crf(emissions, target_tags, input_masks.byte())
        return loss

    def decode(self, emissions, input_masks):
        return self.crf.decode(emissions, input_masks.byte())

    def predict(self, input_ids, input_masks, flags=None, bounds=None, extra=None):
        emissions, *_ = self.tag_outputs(input_ids, input_masks, flags=flags, bounds=bounds, extra=extra)
        return self.decode(emissions, input_masks)


class BERTOnly_Pretrained(nn.Module):
    def __init__(self, pretrained_model_path, max_seq_len, drop_rate, bert_dim,
                 rescale:bool=False, need_flags:bool=False, num_tag=5,
                 need_bounds:bool=False, need_birnn:bool=False, rnn="LSTM", rnn_dim=0,
                 need_extra:bool=False, num_extra=0, **kwargs):
        super(BERTOnly_Pretrained, self).__init__()
        self.max_seq_len = max_seq_len
        self.drop_rate = drop_rate
        self.bert_dim = bert_dim
        self.need_flags = need_flags
        self.need_bounds = need_bounds
        self.need_birnn = need_birnn
        self.need_extra = need_extra
        self.num_tag = num_tag

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
        if need_extra:
            out_dim += num_extra

        self.hidden2tags = nn.Linear(out_dim, num_tag)

    def tag_outputs(self, input_ids, input_masks,
                    flags=None, bounds=None, extra=None):
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
        if self.need_extra:
            outputs = torch.cat([outputs, extra], -1)
        emissions = self.hidden2tags(outputs)
        return emissions

    def forward(self, input_ids, input_masks, target_tags,
                flags=None, bounds=None, extra=None):
        emissions = self.tag_outputs(input_ids, input_masks,
                                     flags=flags, bounds=bounds, extra=extra)
        # scores = self.crf(emissions, target_tags, input_masks.byte())
        loss_fct = nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')
        target_tags = target_tags + (input_masks - 1).long()
        scores = -loss_fct(emissions.view(-1, self.num_tag), target_tags.view(-1))
        return scores

    def decode(self, emissions, input_masks):
        # return self.crf.decode(emissions, input_masks.byte())
        preds = torch.argmax(emissions.detach(), dim=-1)
        preds.masked_fill_(input_masks == 0, 0)
        return preds.cpu().numpy()

    def predict(self, input_ids, input_masks,
                flags=None, bounds=None, extra=None):
        emissions = self.tag_outputs(input_ids, input_masks,
                                     flags=flags, bounds=bounds, extra=extra)
        return self.decode(emissions, input_masks)

