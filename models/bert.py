import math
import sys
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
from pytorch_transformers.modeling_bert import *



@add_start_docstrings("""Bert Model with a `language modeling` head on top. """,
    BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertForMaskedLM_V2(BertPreTrainedModel):
    r"""
        **masked_lm_labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-1, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-1`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``masked_lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Masked language modeling loss.
        **prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, masked_lm_labels=input_ids)
        loss, prediction_scores = outputs[:2]

    """
    def __init__(self, config):
        super(BertForMaskedLM_V2, self).__init__(config)

        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.cls.predictions.decoder,
                                   self.bert.embeddings.word_embeddings)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        outputs = (sequence_output, )  # + outputs[2:]  # Add hidden states and attention if they are here
        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            outputs = outputs + (masked_lm_loss,)
        else:
            outputs = outputs + (0,)

        return outputs  # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)




POS_FLAGS = ['[PAD]', '[CLS]', '[SEP]',
             'ag', 'a', 'ad', 'an', 'b', 'c', 'dg',
             'd', 'e', 'eng', 'f', 'g', 'h', 'i', 'j', 'k',
             'l', 'm', 'ng', 'n', 'nr', 'ns', 'nt', 'nz',
             'o', 'p', 'q', 'r', 's', 'tg', 't', 'u', 'un',
             'vg', 'v', 'vd', 'vn', 'w', 'x', 'y', 'z']

class BERT_Pretrained(nn.Module):
    def __init__(self, pretrained_model_path, max_seq_len, drop_rate, bert_dim,
                 rescale:bool=False, need_flags:bool=False, adj_act="relu", num_tag=5,
                 need_bounds:bool=False, need_birnn:bool=False, rnn="LSTM", rnn_dim=0,
                 need_extra:bool=False, num_extra=0, lm_task:bool=False, word_seg_task:bool=False, **kwargs):
        super(BERT_Pretrained, self).__init__()
        self.max_seq_len = max_seq_len
        self.drop_rate = drop_rate
        # self.gnn_layers = nn.ModuleList()  # compatable needs
        self.bert_dim = bert_dim
        # self.rescale_ws = nn.ParameterList()  # compatable needs
        # self.rescale_bs = nn.ParameterList()  # compatable needs
        self.need_flags = need_flags
        self.need_bounds = need_bounds  # do not use when do word segmentation task
        self.need_birnn = need_birnn
        self.need_extra = need_extra
        self.num_tag = num_tag
        self.lm_task = lm_task
        self.word_seg_task = word_seg_task
        self.crf = CRF(num_tag, batch_first=True)

        if lm_task:
            self.bert4pretrain = BertForMaskedLM_V2.from_pretrained(pretrained_model_path)
        else:
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
        extra_loss = 0
        if self.lm_task:
            seq_outputs, lm_loss = self.bert4pretrain(input_ids, attention_mask=input_masks, 
                                                masked_lm_labels=lm_ids)
            extra_loss = extra_loss + lm_loss
        else:
            seq_outputs, _ = self.bert4pretrain(input_ids, attention_mask=input_masks)

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
                 rescale:bool=False, need_flags:bool=False, adj_act="relu", num_tag=5,
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
