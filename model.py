# -*- coding:utf-8 -*-
# @project: PowerBert
# @filename: model
# @author: 刘聪NLP
# @zhihu: https://www.zhihu.com/people/LiuCongNLP
# @contact: logcongcong@gmail.com
# @time: 2021/9/5 10:14
"""
    文件说明:
            
"""
from transformers import BertModel, BertPreTrainedModel
import torch
import torch.nn as nn
from crf import CRF


class EntityExtractModel(BertPreTrainedModel):
    def __init__(self, config):
        super(EntityExtractModel, self).__init__(config)
        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, attention_mask, labels=None):
        sequence_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        logits = self.classifier(sequence_output)
        tags = self.crf.decode(logits, attention_mask)
        outputs = (tags, logits,)
        if labels is not None:
            loss = self.crf(emissions=logits, tags=labels, mask=attention_mask)
            outputs = (-1 * loss,) + outputs
        return outputs  # (loss), tags, logits


class DefectExtractModel(BertPreTrainedModel):
    def __init__(self, config, num_label1, num_label2, num_label3, num_label4, num_label5):
        super(DefectExtractModel, self).__init__(config)
        self.bert = BertModel(config)
        self.classifier1 = nn.Linear(config.hidden_size, num_label1)
        self.classifier2 = nn.Linear(config.hidden_size, num_label2)
        self.classifier3 = nn.Linear(config.hidden_size, num_label3)
        self.classifier4 = nn.Linear(config.hidden_size, num_label4)
        self.qa_outputs = nn.Linear(config.hidden_size, num_label5)
        self.init_weights()

    def forward(self, input_ids, attention_mask, label1=None, label2=None, label3=None, label4=None, label5_start=None,
                label5_end=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        pooled_output = outputs[1]
        logit1 = self.classifier1(pooled_output)
        predict_label1 = torch.argmax(logit1, dim=-1)
        logit2 = self.classifier2(pooled_output)
        predict_label2 = torch.argmax(logit2, dim=-1)
        logit3 = self.classifier3(pooled_output)
        predict_label3 = torch.argmax(logit3, dim=-1)
        logit4 = self.classifier4(pooled_output)
        predict_label4 = torch.argmax(logit4, dim=-1)
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        pre_start_label = torch.argmax(start_logits, dim=-1)
        pre_end_label = torch.argmax(end_logits, dim=-1)
        outputs = (
            predict_label1, logit1, predict_label2, logit2, predict_label3, logit3, predict_label4, logit4,
            pre_start_label, start_logits, pre_end_label, end_logits)
        if label1 is not None and label2 is not None and label3 is not None and label4 is not None and label5_start is not None and label5_end is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss1 = loss_fct(logit1, label1)
            loss2 = loss_fct(logit2, label2)
            loss3 = loss_fct(logit3, label3)
            loss4 = loss_fct(logit4, label4)
            start_loss = loss_fct(start_logits, label5_start)
            end_loss = loss_fct(end_logits, label5_end)
            loss5 = (start_loss + end_loss) / 2
            total_loss = (loss1 + loss2 + loss3 + loss4) / 4
            outputs = (total_loss, loss1, loss2, loss3, loss4, loss5) + outputs
        return outputs  # (loss), tags, logits


class DefectDiagnosisModel(BertPreTrainedModel):
    def __init__(self, config):
        super(DefectDiagnosisModel, self).__init__(config)
        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids, attention_mask, label=None):
        pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[1]
        logits = self.classifier(pooled_output)
        predict_label = torch.argmax(logits, dim=-1)
        outputs = (predict_label, logits,)
        if label is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, label)
            outputs = (loss,) + outputs
        return outputs  # (loss), predict_label, logits
