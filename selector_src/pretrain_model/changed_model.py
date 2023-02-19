import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import math

from modeling_bert import BertPreTrainedModel, BertModel, BertOutput, BertSelfOutput, BertIntermediate
from transformers import RobertaModel, ElectraModel, AlbertModel
from transformer import TransformerLayer


def gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def swish(x):
    return x * torch.sigmoid(x)


def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "gelu_new": gelu_new, "mish": mish}


class BertForParagraphClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForParagraphClassification, self).__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                cls_mask=None,
                pq_end_pos=None,
                cls_label=None,
                cls_weight=None):
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            cls_mask = cls_mask.unsqueeze(0)
            if pq_end_pos is not None:
                pq_end_pos = cls_mask.unsqueeze(0)
            if cls_label is not None:
                cls_label = cls_label.unsqueeze(0)
                cls_weight = cls_weight.unsqueeze(0)
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        cls_output = outputs[1]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output).squeeze(-1)
        if cls_label is None:
            return logits
        # 选择第一个标记结果
        cls_label = torch.index_select(cls_label, dim=1, index=torch.tensor([0, ]).cuda())
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.config.num_labels), cls_label.view(-1))
        return loss, logits  # (loss), scores, (hidden_states), (attentions)


class BertForParagraphClassificationMean(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForParagraphClassificationMean, self).__init__(config)

        self.bert = BertModel(config)
        self.avg_pool = nn.AvgPool1d(512)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                cls_mask=None,
                pq_end_pos=None,
                cls_label=None,
                cls_weight=None):
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            cls_mask = cls_mask.unsqueeze(0)
            if pq_end_pos is not None:
                pq_end_pos = cls_mask.unsqueeze(0)
            if cls_label is not None:
                cls_label = cls_label.unsqueeze(0)
                cls_weight = cls_weight.unsqueeze(0)
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds)
        outputs = outputs[0]
        cls_output = self.avg_pool(outputs.permute(0, 2, 1)).permute(0, 2, 1).squeeze(1)
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output).squeeze(-1)
        if cls_label is None:
            return logits
        # 选择第一个标记结果
        cls_label = torch.index_select(cls_label, dim=1, index=torch.tensor([0, ]).cuda())
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.config.num_labels), cls_label.view(-1))
        return loss, logits  # (loss), scores, (hidden_states), (attentions)


class BertForParagraphClassificationMax(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForParagraphClassificationMax, self).__init__(config)

        self.bert = BertModel(config)
        self.max_pool = nn.MaxPool1d(512)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                cls_mask=None,
                pq_end_pos=None,
                cls_label=None,
                cls_weight=None):
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            cls_mask = cls_mask.unsqueeze(0)
            if pq_end_pos is not None:
                pq_end_pos = cls_mask.unsqueeze(0)
            if cls_label is not None:
                cls_label = cls_label.unsqueeze(0)
                cls_weight = cls_weight.unsqueeze(0)
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds)
        outputs = outputs[0]
        cls_output = self.max_pool(outputs.permute(0, 2, 1)).permute(0, 2, 1).squeeze(1)
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output).squeeze(-1)
        if cls_label is None:
            return logits
        # 选择第一个标记结果
        cls_label = torch.index_select(cls_label, dim=1, index=torch.tensor([0, ]).cuda())
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.config.num_labels), cls_label.view(-1))
        return loss, logits  # (loss), scores, (hidden_states), (attentions)


class BertForRelatedSentence(BertPreTrainedModel):

    def __init__(self, config):
        super(BertForRelatedSentence, self).__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                cls_mask=None,
                pq_end_pos=None,
                cls_label=None,
                cls_weight=None):
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            if pq_end_pos is not None:
                pq_end_pos = cls_mask.unsqueeze(0)
            if cls_label is not None:
                cls_label = cls_label.unsqueeze(0)
                cls_mask = cls_mask.unsqueeze(0)
                cls_weight = cls_weight.unsqueeze(0)
        sequence_output, _ = self.bert(input_ids,
                                       attention_mask=attention_mask,
                                       token_type_ids=token_type_ids,
                                       position_ids=position_ids,
                                       head_mask=head_mask,
                                       inputs_embeds=inputs_embeds)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output).squeeze(-1)
        loss_fn1 = torch.nn.BCEWithLogitsLoss(reduce=False, size_average=False)
        logits = logits * cls_mask.float()
        if cls_label is None:
            return logits
        loss1 = loss_fn1(logits, cls_label.float())
        weighted_loss1 = (loss1 * cls_mask.float()) * cls_weight
        loss1 = torch.sum(weighted_loss1, (-1, -2), keepdim=False)
        logits = torch.sigmoid(logits)
        return loss1, logits


class ElectraForParagraphClassification(ElectraModel):
    def __init__(self, config):
        super(ElectraForParagraphClassification, self).__init__(config)

        self.electra = ElectraModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                cls_mask=None,
                pq_end_pos=None,
                cls_label=None,
                cls_weight=None):
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            cls_mask = cls_mask.unsqueeze(0)
            if pq_end_pos is not None:
                pq_end_pos = cls_mask.unsqueeze(0)
            if cls_label is not None:
                cls_label = cls_label.unsqueeze(0)
                cls_weight = cls_weight.unsqueeze(0)
        outputs = self.electra(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask,
                               inputs_embeds=inputs_embeds)
        cls_output = outputs[0]
        cls_output = torch.index_select(cls_output, dim=1, index=torch.tensor([0, ]).cuda())
        cls_output = cls_output.squeeze(1)
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output).squeeze(-1)
        if cls_label is None:
            return logits
        # 选择第一个标记结果
        cls_label = torch.index_select(cls_label, dim=1, index=torch.tensor([0, ]).cuda())
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.config.num_labels), cls_label.view(-1))
        return loss, logits  # (loss), scores, (hidden_states), (attentions)


class AlbertForParagraphClassification(AlbertModel):
    def __init__(self, config):
        super(AlbertForParagraphClassification, self).__init__(config)

        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                cls_mask=None,
                pq_end_pos=None,
                cls_label=None,
                cls_weight=None):
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            cls_mask = cls_mask.unsqueeze(0)
            if pq_end_pos is not None:
                pq_end_pos = cls_mask.unsqueeze(0)
            if cls_label is not None:
                cls_label = cls_label.unsqueeze(0)
                cls_weight = cls_weight.unsqueeze(0)
        outputs = self.albert(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask,
                               inputs_embeds=inputs_embeds)
        cls_output = outputs[0]
        cls_output = torch.index_select(cls_output, dim=1, index=torch.tensor([0, ]).cuda())
        cls_output = cls_output.squeeze(1)
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output).squeeze(-1)
        if cls_label is None:
            return logits
        # 选择第一个标记结果
        cls_label = torch.index_select(cls_label, dim=1, index=torch.tensor([0, ]).cuda())
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.config.num_labels), cls_label.view(-1))
        return loss, logits  # (loss), scores, (hidden_states), (attentions)


class ElectraForParagraphClassificationCrossAttention(ElectraModel):
    def __init__(self, config):
        super(ElectraForParagraphClassificationCrossAttention, self).__init__(config)

        self.electra = ElectraModel(config)
        self.cross_attention = CrossAttention(config=config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                cls_mask=None,
                pq_end_pos=None,
                cls_label=None,
                cls_weight=None):
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            cls_mask = cls_mask.unsqueeze(0)
            if pq_end_pos is not None:
                pq_end_pos = pq_end_pos.unsqueeze(0)
            if cls_label is not None:
                cls_label = cls_label.unsqueeze(0)
                cls_weight = cls_weight.unsqueeze(0)
        outputs = self.electra(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask,
                               inputs_embeds=inputs_embeds)

        sequence_output = outputs[0]
        cls_output = torch.index_select(sequence_output, dim=1, index=torch.tensor([0, ]).cuda())
        cls_output = cls_output.squeeze(1)
        query_sequence_output, context_sequence_output, query_attention_mask, context_attention_mask = \
            split_ques_context(sequence_output, pq_end_pos)
        sequence_output = self.cross_attention(query_sequence_output, sequence_output, query_attention_mask)
        sequence_output = sequence_output + outputs[0]
        cross_cls_output = torch.index_select(sequence_output, dim=1, index=torch.tensor([0, ]).cuda())
        cross_cls_output = cross_cls_output.squeeze(1)
        cls_output = cls_output + cross_cls_output
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output).squeeze(-1)
        if cls_label is None:
            return logits
        # 选择第一个标记结果
        cls_label = torch.index_select(cls_label, dim=1, index=torch.tensor([0, ]).cuda())
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.config.num_labels), cls_label.view(-1))
        return loss, logits  # (loss), scores, (hidden_states), (attentions)


class ElectraSecondForParagraphClassificationCrossAttention(ElectraModel):
    def __init__(self, config):
        super(ElectraSecondForParagraphClassificationCrossAttention, self).__init__(config)

        self.electra = ElectraModel(config)
        self.cross_attention = CrossAttention(config=config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                cls_mask=None,
                pq_end_pos=None,
                cls_label=None,
                cls_weight=None):
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            cls_mask = cls_mask.unsqueeze(0)
            if pq_end_pos is not None:
                pq_end_pos = pq_end_pos.unsqueeze(0)
            if cls_label is not None:
                cls_label = cls_label.unsqueeze(0)
                cls_weight = cls_weight.unsqueeze(0)
        outputs = self.electra(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask,
                               inputs_embeds=inputs_embeds)

        sequence_output = outputs[0]
        cls_output = torch.index_select(sequence_output, dim=1, index=torch.tensor([0, ]).cuda())
        cls_output = cls_output.squeeze(1)
        query_sequence_output, context_sequence_output, query_attention_mask, context_attention_mask = \
            split_ques_context2(sequence_output, pq_end_pos)
        sequence_output = self.cross_attention(query_sequence_output, sequence_output, query_attention_mask)
        sequence_output = sequence_output + outputs[0]
        cross_cls_output = torch.index_select(sequence_output, dim=1, index=torch.tensor([0, ]).cuda())
        cross_cls_output = cross_cls_output.squeeze(1)
        cls_output = cls_output + cross_cls_output
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output).squeeze(-1)
        if cls_label is None:
            return logits
        # 选择第一个标记结果
        cls_label = torch.index_select(cls_label, dim=1, index=torch.tensor([0, ]).cuda())
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.config.num_labels), cls_label.view(-1))
        return loss, logits  # (loss), scores, (hidden_states), (attentions)


class AlbertSecondForParagraphClassificationCrossAttention(AlbertModel):
    def __init__(self, config):
        super(AlbertSecondForParagraphClassificationCrossAttention, self).__init__(config)

        self.albert = AlbertModel(config)
        self.cross_attention = CrossAttention(config=config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                cls_mask=None,
                pq_end_pos=None,
                cls_label=None,
                cls_weight=None):
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            cls_mask = cls_mask.unsqueeze(0)
            if pq_end_pos is not None:
                pq_end_pos = pq_end_pos.unsqueeze(0)
            if cls_label is not None:
                cls_label = cls_label.unsqueeze(0)
                cls_weight = cls_weight.unsqueeze(0)
        outputs = self.albert(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask,
                               inputs_embeds=inputs_embeds)

        sequence_output = outputs[0]
        cls_output = torch.index_select(sequence_output, dim=1, index=torch.tensor([0, ]).cuda())
        cls_output = cls_output.squeeze(1)
        query_sequence_output, context_sequence_output, query_attention_mask, context_attention_mask = \
            split_ques_context2(sequence_output, pq_end_pos)
        sequence_output = self.cross_attention(query_sequence_output, sequence_output, query_attention_mask)
        sequence_output = sequence_output + outputs[0]
        cross_cls_output = torch.index_select(sequence_output, dim=1, index=torch.tensor([0, ]).cuda())
        cross_cls_output = cross_cls_output.squeeze(1)
        cls_output = cls_output + cross_cls_output
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output).squeeze(-1)
        if cls_label is None:
            return logits
        # 选择第一个标记结果
        cls_label = torch.index_select(cls_label, dim=1, index=torch.tensor([0, ]).cuda())
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.config.num_labels), cls_label.view(-1))
        return loss, logits  # (loss), scores, (hidden_states), (attentions)


class ElectraForRelatedSentence(ElectraModel):

    def __init__(self, config):
        super(ElectraForRelatedSentence, self).__init__(config)

        self.electra = ElectraModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                cls_mask=None,
                pq_end_pos=None,
                cls_label=None,
                cls_weight=None):
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            if pq_end_pos is not None:
                pq_end_pos = cls_mask.unsqueeze(0)
            if cls_label is not None:
                cls_label = cls_label.unsqueeze(0)
                cls_mask = cls_mask.unsqueeze(0)
                cls_weight = cls_weight.unsqueeze(0)
        sequence_output = self.electra(input_ids,
                                       attention_mask=attention_mask,
                                       token_type_ids=token_type_ids,
                                       position_ids=position_ids,
                                       head_mask=head_mask,
                                       inputs_embeds=inputs_embeds)
        sequence_output = sequence_output[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output).squeeze(-1)
        loss_fn1 = torch.nn.BCEWithLogitsLoss(reduce=False, size_average=False)
        logits = logits * cls_mask.float()
        if cls_label is None:
            return logits
        loss1 = loss_fn1(logits, cls_label.float())
        weighted_loss1 = (loss1 * cls_mask.float()) * cls_weight
        loss1 = torch.sum(weighted_loss1, (-1, -2), keepdim=False)
        logits = torch.sigmoid(logits)
        return loss1, logits


class RobertaForParagraphClassification(RobertaModel):
    def __init__(self, config):
        super(RobertaForParagraphClassification, self).__init__(config)

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                cls_mask=None,
                pq_end_pos=None,
                cls_label=None,
                cls_weight=None):
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            cls_mask = cls_mask.unsqueeze(0)
            if pq_end_pos is not None:
                pq_end_pos = cls_mask.unsqueeze(0)
            if cls_label is not None:
                cls_label = cls_label.unsqueeze(0)
                cls_weight = cls_weight.unsqueeze(0)
        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               position_ids=position_ids,
                               head_mask=head_mask,
                               inputs_embeds=inputs_embeds)
        cls_output = outputs[0]
        cls_output = torch.index_select(cls_output, dim=1, index=torch.tensor([0, ]).cuda())
        cls_output = cls_output.squeeze(1)
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output).squeeze(-1)
        if cls_label is None:
            return logits
        # 选择第一个标记结果
        cls_label = torch.index_select(cls_label, dim=1, index=torch.tensor([0, ]).cuda())
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.config.num_labels), cls_label.view(-1))
        return loss, logits  # (loss), scores, (hidden_states), (attentions)


class RobertaForRelatedSentence(RobertaModel):

    def __init__(self, config):
        super(RobertaForRelatedSentence, self).__init__(config)

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                cls_mask=None,
                pq_end_pos=None,
                cls_label=None,
                cls_weight=None):
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            if pq_end_pos is not None:
                pq_end_pos = cls_mask.unsqueeze(0)
            if cls_label is not None:
                cls_label = cls_label.unsqueeze(0)
                cls_mask = cls_mask.unsqueeze(0)
                cls_weight = cls_weight.unsqueeze(0)
        sequence_output = self.roberta(input_ids,
                                       attention_mask=attention_mask,
                                       position_ids=position_ids,
                                       head_mask=head_mask,
                                       inputs_embeds=inputs_embeds)
        sequence_output = sequence_output[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output).squeeze(-1)
        loss_fn1 = torch.nn.BCEWithLogitsLoss(reduce=False, size_average=False)
        logits = logits * cls_mask.float()
        if cls_label is None:
            return logits
        loss1 = loss_fn1(logits, cls_label.float())
        weighted_loss1 = (loss1 * cls_mask.float()) * cls_weight
        loss1 = torch.sum(weighted_loss1, (-1, -2), keepdim=False)
        logits = torch.sigmoid(logits)
        return loss1, logits


class CrossAttention(nn.Module):
    def __init__(self, config):
        super(CrossAttention, self).__init__()

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))

        self.output_attentions = config.output_attentions
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.attention_head_size = config.hidden_size // config.num_attention_heads

        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pruned_heads = set()

        self.full_layer_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ffn = nn.Linear(config.hidden_size, config.intermediate_size)
        self.ffn_output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation = ACT2FN[config.hidden_act]

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    # attention mask 对应 input_ids
    def forward(self, input_ids, input_ids_1, attention_mask=None, head_mask=None):
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        attention_mask = extended_attention_mask

        mixed_query_layer = self.query(input_ids_1)
        mixed_key_layer = self.key(input_ids)
        mixed_value_layer = self.value(input_ids)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        reshaped_context_layer = context_layer.view(*new_context_layer_shape)

        # Should find a better way to do this
        w = self.dense.weight.t().view(self.num_attention_heads, self.attention_head_size, self.hidden_size).to(
            context_layer.dtype)
        b = self.dense.bias.to(context_layer.dtype)

        projected_context_layer = torch.einsum("bfnd,ndh->bfh", context_layer, w) + b
        projected_context_layer_dropout = self.dropout(projected_context_layer)
        layernormed_context_layer = self.LayerNorm(input_ids_1 + projected_context_layer_dropout)

        ffn_output = self.ffn(layernormed_context_layer)
        ffn_output = self.activation(ffn_output)
        ffn_output = self.ffn_output(ffn_output)
        hidden_states = self.full_layer_layer_norm(ffn_output + layernormed_context_layer)
        return hidden_states


def split_ques_context(sequence_output, pq_end_pos):
    ques_max_len = 64
    context_max_len = 512 - 64
    sep_tok_len = 1
    ques_sequence_output = sequence_output.new(
        torch.Size((sequence_output.size(0), ques_max_len, sequence_output.size(2)))).zero_()
    context_sequence_output = sequence_output.new_zeros(
        (sequence_output.size(0), context_max_len, sequence_output.size(2)))
    context_attention_mask = sequence_output.new_zeros((sequence_output.size(0), context_max_len))
    ques_attention_mask = sequence_output.new_zeros((sequence_output.size(0), ques_max_len))
    for i in range(0, sequence_output.size(0)):
        q_end = pq_end_pos[i][0]
        p_end = pq_end_pos[i][1]
        ques_sequence_output[i, :min(ques_max_len, q_end)] = sequence_output[i,
                                                             1: 1 + min(ques_max_len, q_end)]
        context_sequence_output[i, :min(context_max_len, p_end - q_end - sep_tok_len)] = sequence_output[i,
                                                                                         q_end + sep_tok_len + 1: q_end + sep_tok_len + 1 + min(
                                                                                             p_end - q_end - sep_tok_len,
                                                                                             context_max_len)]
        context_attention_mask[i, :min(context_max_len, p_end - q_end - sep_tok_len)] = sequence_output.new_ones(
            (1, context_max_len))[0, :min(context_max_len, p_end - q_end - sep_tok_len)]
        ques_attention_mask[i, : min(ques_max_len, q_end)] = sequence_output.new_ones((1, ques_max_len))[0,
                                                             : min(ques_max_len, q_end)]
    return ques_sequence_output, context_sequence_output, ques_attention_mask, context_attention_mask


def split_ques_context2(sequence_output, pq_end_pos):
    ques_max_len = 256
    context_max_len = 512 - ques_max_len
    sep_tok_len = 1
    ques_sequence_output = sequence_output.new(
        torch.Size((sequence_output.size(0), ques_max_len, sequence_output.size(2)))).zero_()
    context_sequence_output = sequence_output.new_zeros(
        (sequence_output.size(0), context_max_len, sequence_output.size(2)))
    context_attention_mask = sequence_output.new_zeros((sequence_output.size(0), context_max_len))
    ques_attention_mask = sequence_output.new_zeros((sequence_output.size(0), ques_max_len))
    for i in range(0, sequence_output.size(0)):
        q_end = pq_end_pos[i][0]
        p_end = pq_end_pos[i][1]
        ques_sequence_output[i, :min(ques_max_len, q_end)] = sequence_output[i,
                                                             1: 1 + min(ques_max_len, q_end)]
        context_sequence_output[i, :min(context_max_len, p_end - q_end - sep_tok_len)] = sequence_output[i,
                                                                                         q_end + sep_tok_len + 1: q_end + sep_tok_len + 1 + min(
                                                                                             p_end - q_end - sep_tok_len,
                                                                                             context_max_len)]
        context_attention_mask[i, :min(context_max_len, p_end - q_end - sep_tok_len)] = sequence_output.new_ones(
            (1, context_max_len))[0, :min(context_max_len, p_end - q_end - sep_tok_len)]
        ques_attention_mask[i, : min(ques_max_len, q_end)] = sequence_output.new_ones((1, ques_max_len))[0,
                                                             : min(ques_max_len, q_end)]
    return ques_sequence_output, context_sequence_output, ques_attention_mask, context_attention_mask


class BertForRelatedSentenceWithCrossAttention(BertPreTrainedModel):

    def __init__(self, config):
        super(BertForRelatedSentenceWithCrossAttention, self).__init__(config)
        self.bert = BertModel(config)
        self.cross_attention = CrossAttention(config=config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                cls_mask=None,
                pq_end_pos=None,
                cls_label=None,
                cls_weight=None):
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        query_sequence_output, context_sequence_output, query_attention_mask, context_attention_mask = \
            split_ques_context(sequence_output, pq_end_pos)
        sequence_output = self.cross_attention(query_sequence_output, sequence_output, query_attention_mask)
        sequence_output = sequence_output + outputs[0]
        logits = self.classifier(sequence_output).squeeze(-1)
        loss_fn1 = torch.nn.BCEWithLogitsLoss(reduce=False, size_average=False)
        logits = logits * cls_mask.float()
        if cls_label is None:
            return logits
        loss1 = loss_fn1(logits, cls_label.float())
        weighted_loss1 = (loss1 * cls_mask.float()) * cls_weight
        loss1 = torch.sum(weighted_loss1, (-1, -2), keepdim=False)
        logits = torch.sigmoid(logits)
        return loss1, logits


class BertCoattention(nn.Module):
    def __init__(self, config):
        super(BertCoattention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, doc_states, que_states, attention_mask):
        mixed_query_layer = self.query(doc_states)  # b*doc*all
        mixed_key_layer = self.key(que_states)  # b*que*all
        mixed_value_layer = self.value(que_states)  # b*que*all

        query_layer = self.transpose_for_scores(mixed_query_layer)  # b*num_head*doc*head_size
        key_layer = self.transpose_for_scores(mixed_key_layer)  # b*num_head*que*head_size
        value_layer = self.transpose_for_scores(mixed_value_layer)  # b*num_head*que*head_size

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # b*num_head*doc*que
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)  # b*num_head*doc*head_size
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # b*doc*num_head*head_size
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)  # b*doc*all
        return context_layer


class BertCoAttention(nn.Module):
    def __init__(self, config):
        super(BertCoAttention, self).__init__()
        self.self = BertCoattention(config)
        self.output = BertSelfOutput(config)

    def forward(self, doc_tensor, que_tensor, attention_mask):
        self_output = self.self(doc_tensor, que_tensor, attention_mask)
        attention_output = self.output(self_output, doc_tensor)
        return attention_output


class BertCoLayer(nn.Module):
    def __init__(self, config):
        super(BertCoLayer, self).__init__()
        self.attention = BertCoAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, doc_states, que_states, attention_mask):
        attention_output = self.attention(doc_states, que_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class CoattentionModel(BertPreTrainedModel):
    def __init__(self, config):
        super(CoattentionModel, self).__init__(config)
        self.encoder = BertCoLayer(config)
        self.init_weights()

    def forward(self, doc_embedding, que_embedding, attention_mask):
        # attention_mask b*doc*que
        extended_attention_mask = attention_mask.unsqueeze(1)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        encoded_layer = self.encoder(doc_embedding, que_embedding, extended_attention_mask)
        # pooled_output = self.pooler(encoded_layer)
        return encoded_layer


class BertForQuestionAnsweringForward(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForQuestionAnsweringForward, self).__init__(config)
        self.bert = BertModel(config)

        self.start_logits = nn.Linear(config.hidden_size, 1)
        self.end_logits = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sent = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids,
                word_sim_matrix=None,
                start_positions=None,
                end_positions=None,
                sent_mask=None,
                sent_lbs=None,
                sent_weight=None):
        if len(input_ids.shape) < 2:
            input_ids = input_ids.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            if start_positions is not None and len(start_positions.shape) < 2:
                start_positions = start_positions.unsqueeze(0)
                end_positions = end_positions.unsqueeze(0)
                sent_mask = sent_mask.unsqueeze(0)
                sent_lbs = sent_lbs.unsqueeze(0)
                sent_weight = sent_weight.unsqueeze(0)
        sequence_output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        start_logits = self.start_logits(sequence_output).squeeze(-1)  # +extended_context_mask#*context_mask.float()
        end_logits = self.end_logits(sequence_output).squeeze(-1)  # +extended_context_mask#*context_mask.float()

        sent_logits = self.sent(sequence_output).squeeze(-1)  # *context_mask.float()
        if len(sent_logits) > 1:
            sent_logits.squeeze(-1)
        loss_fn1 = torch.nn.BCEWithLogitsLoss(reduce=False, size_average=False)
        sent_logits = sent_logits  # * sent_mask.float()
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            sent_loss = loss_fn1(sent_logits, sent_lbs.float())
            sent_loss = torch.sum(sent_loss, (-1, -2), keepdim=False)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)  # bsz*seq bsz*n
            end_loss = loss_fct(end_logits, end_positions)
            ans_loss = start_loss + end_loss
            total_loss = ans_loss + 0.2 * sent_loss
            return total_loss, start_logits, end_logits, sent_logits
        else:
            start_logits = nn.Softmax(dim=-1)(start_logits)
            end_logits = nn.Softmax(dim=-1)(end_logits)
            sent_logits = torch.sigmoid(sent_logits)
            return start_logits, end_logits, sent_logits


class BertForQuestionAnsweringForwardBest(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForQuestionAnsweringForwardBest, self).__init__(config)
        self.bert = BertModel(config)
        self.start_logits = nn.Linear(config.hidden_size, 1)
        self.end_logits = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sent = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids,
                word_sim_matrix=None,
                start_positions=None,
                end_positions=None,
                sent_mask=None,
                sent_lbs=None,
                sent_weight=None):
        if len(input_ids.shape) < 2:
            input_ids = input_ids.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            if start_positions is not None and len(start_positions.shape) < 2:
                start_positions = start_positions.unsqueeze(0)
                end_positions = end_positions.unsqueeze(0)
                sent_mask = sent_mask.unsqueeze(0)
                sent_lbs = sent_lbs.unsqueeze(0)
                sent_weight = sent_weight.unsqueeze(0)
        sequence_output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        sequence_output = self.dropout(sequence_output)
        ones_mask = torch.ones_like(attention_mask).cuda()
        context_mask = (ones_mask - token_type_ids) * attention_mask
        extended_context_mask = (1.0 - context_mask) * -10000.0
        start_logits = self.start_logits(sequence_output).squeeze(-1) + extended_context_mask  # *context_mask.float()
        end_logits = self.end_logits(sequence_output).squeeze(-1) + extended_context_mask  # *context_mask.float()
        # 去除context mask
        sent_logits = self.sent(sequence_output).squeeze(-1) * context_mask.float()
        # sent_logits = self.sent(sequence_output).squeeze(-1)
        if len(sent_logits) > 1:
            sent_logits.squeeze(-1)
        loss_fn1 = torch.nn.BCEWithLogitsLoss(reduce=False, size_average=False)
        # 去除sent_mask
        sent_logits = sent_logits * sent_mask.float()
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sent_lbs = sent_lbs[:, 0:context_maxlen]
            # sent_weight = sent_weight[:, 0:context_maxlen]
            sent_loss = loss_fn1(sent_logits, sent_lbs.float())
            sent_loss = (sent_loss * sent_mask.float()) * sent_weight
            # sent_loss = (sent_loss * sent_mask.float())
            sent_loss = torch.sum(sent_loss, (-1, -2), keepdim=False)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)  # bsz*seq bsz*n
            end_loss = loss_fct(end_logits, end_positions)
            ans_loss = start_loss + end_loss
            total_loss = ans_loss + 0.2 * sent_loss
            return total_loss, start_logits, end_logits, sent_logits
        else:
            start_logits = nn.Softmax(dim=-1)(start_logits)
            end_logits = nn.Softmax(dim=-1)(end_logits)
            sent_logits = torch.sigmoid(sent_logits)
            return start_logits, end_logits, sent_logits


class BertTransformer(BertPreTrainedModel):
    def __init__(self, config):
        super(BertTransformer, self).__init__(config)
        self.bert = BertModel(config)
        self.transformer = TransformerLayer(hidden_size=config.hidden_size,
                                            head_num=24,
                                            dropout=config.hidden_dropout_prob,
                                            feedforward_size=config.hidden_size * 3)
        self.start_logits = nn.Linear(config.hidden_size, 1)
        self.end_logits = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)
        self.sent = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids,
                word_sim_matrix=None,
                start_positions=None,
                end_positions=None,
                sent_mask=None,
                sent_lbs=None,
                sent_weight=None):
        if len(input_ids.shape) < 2:
            input_ids = input_ids.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            if start_positions is not None and len(start_positions.shape) < 2:
                start_positions = start_positions.unsqueeze(0)
                end_positions = end_positions.unsqueeze(0)
                sent_mask = sent_mask.unsqueeze(0)
                sent_lbs = sent_lbs.unsqueeze(0)
                sent_weight = sent_weight.unsqueeze(0)
        sequence_output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        batch_size, seq_length, hidden_size = sequence_output.size()
        sequence_output = self.dropout(sequence_output)
        self_mask = attention_mask.unsqueeze(1).repeat(1, seq_length, 1).unsqueeze(1)
        word_sim_matrix = word_sim_matrix.unsqueeze(dim=1)
        sequence_output = self.transformer(sequence_output, mask=self_mask + word_sim_matrix)
        sequence_output = self.dropout(sequence_output)
        ones_mask = torch.ones_like(attention_mask).cuda()
        context_mask = (ones_mask - token_type_ids) * attention_mask
        ques_mask = token_type_ids * attention_mask
        extended_context_mask = (1.0 - context_mask) * -10000.0
        start_logits = self.start_logits(sequence_output).squeeze(-1) + extended_context_mask  # *context_mask.float()
        end_logits = self.end_logits(sequence_output).squeeze(-1) + extended_context_mask  # *context_mask.float()

        sent_logits = self.sent(sequence_output).squeeze(-1) * context_mask.float()
        if len(sent_logits) > 1:
            sent_logits.squeeze(-1)
        loss_fn1 = torch.nn.BCEWithLogitsLoss(reduce=False, size_average=False)
        sent_logits = sent_logits * sent_mask.float()
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            sent_loss = loss_fn1(sent_logits, sent_lbs.float())
            sent_loss = (sent_loss * sent_mask.float()) * sent_weight
            sent_loss = torch.sum(sent_loss, (-1, -2), keepdim=False)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)  # bsz*seq bsz*n
            end_loss = loss_fct(end_logits, end_positions)
            ans_loss = start_loss + end_loss
            total_loss = ans_loss + 0.2 * sent_loss
            return total_loss, start_logits, end_logits, sent_logits
        else:
            start_logits = nn.Softmax(dim=-1)(start_logits)
            end_logits = nn.Softmax(dim=-1)(end_logits)
            sent_logits = torch.sigmoid(sent_logits)
            return start_logits, end_logits, sent_logits


class BertSkipConnectTransformer(BertPreTrainedModel):
    def __init__(self, config):
        super(BertSkipConnectTransformer, self).__init__(config)
        self.bert = BertModel(config)
        self.transformer = TransformerLayer(hidden_size=config.hidden_size,
                                            head_num=12,
                                            dropout=config.hidden_dropout_prob,
                                            feedforward_size=config.hidden_size * 3)
        self.start_logits = nn.Linear(config.hidden_size, 1)
        self.end_logits = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sent = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids,
                start_positions=None,
                end_positions=None,
                sent_mask=None,
                sent_lbs=None,
                sent_weight=None):
        if len(input_ids.shape) < 2:
            input_ids = input_ids.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            if start_positions is not None and len(start_positions.shape) < 2:
                start_positions = start_positions.unsqueeze(0)
                end_positions = end_positions.unsqueeze(0)
                sent_mask = sent_mask.unsqueeze(0)
                sent_lbs = sent_lbs.unsqueeze(0)
                sent_weight = sent_weight.unsqueeze(0)
        sequence_output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        batch_size, seq_length, hidden_size = sequence_output.size()
        sequence_output = self.dropout(sequence_output)
        self_mask = attention_mask.unsqueeze(1).repeat(1, seq_length, 1).unsqueeze(1)
        sequence_output = sequence_output + self.transformer(sequence_output, mask=self_mask)
        sequence_output = self.dropout(sequence_output)
        ones_mask = torch.ones_like(attention_mask).cuda()
        context_mask = (ones_mask - token_type_ids) * attention_mask
        ques_mask = token_type_ids * attention_mask
        extended_context_mask = (1.0 - context_mask) * -10000.0
        start_logits = self.start_logits(sequence_output).squeeze(-1) + extended_context_mask  # *context_mask.float()
        end_logits = self.end_logits(sequence_output).squeeze(-1) + extended_context_mask  # *context_mask.float()

        sent_logits = self.sent(sequence_output).squeeze(-1) * context_mask.float()
        if len(sent_logits) > 1:
            sent_logits.squeeze(-1)
        loss_fn1 = torch.nn.BCEWithLogitsLoss(reduce=False, size_average=False)
        sent_logits = sent_logits * sent_mask.float()
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            sent_loss = loss_fn1(sent_logits, sent_lbs.float())
            sent_loss = (sent_loss * sent_mask.float()) * sent_weight
            sent_loss = torch.sum(sent_loss, (-1, -2), keepdim=False)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)  # bsz*seq bsz*n
            end_loss = loss_fct(end_logits, end_positions)
            ans_loss = start_loss + end_loss
            total_loss = ans_loss + 0.2 * sent_loss
            return total_loss, start_logits, end_logits, sent_logits
        else:
            start_logits = nn.Softmax(dim=-1)(start_logits)
            end_logits = nn.Softmax(dim=-1)(end_logits)
            sent_logits = torch.sigmoid(sent_logits)
            return start_logits, end_logits, sent_logits


class BertSelfAttentionAndCoAttention(BertPreTrainedModel):
    def __init__(self, config):
        super(BertSelfAttentionAndCoAttention, self).__init__(config)
        self.bert = BertModel(config)
        self.transformer = TransformerLayer(hidden_size=config.hidden_size,
                                            head_num=12,
                                            dropout=config.hidden_dropout_prob,
                                            feedforward_size=config.hidden_size * 3)
        self.Wq1_1 = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.Wq1_2 = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.coatt1 = CoattentionModel(config)
        self.start_logits = nn.Linear(config.hidden_size, 1)
        self.end_logits = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sent = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids,
                start_positions=None,
                end_positions=None,
                sent_mask=None,
                sent_lbs=None,
                sent_weight=None):
        if len(input_ids.shape) < 2:
            input_ids = input_ids.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            if start_positions is not None and len(start_positions.shape) < 2:
                start_positions = start_positions.unsqueeze(0)
                end_positions = end_positions.unsqueeze(0)
                sent_mask = sent_mask.unsqueeze(0)
                sent_lbs = sent_lbs.unsqueeze(0)
                sent_weight = sent_weight.unsqueeze(0)
        sequence_output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        batch_size, seq_length, hidden_size = sequence_output.size()
        sequence_output = self.dropout(sequence_output)
        self_mask = attention_mask.unsqueeze(1).repeat(1, seq_length, 1).unsqueeze(1)
        sequence_output = self.transformer(sequence_output, mask=self_mask)
        sequence_output = self.dropout(sequence_output)

        ones_mask = torch.ones_like(attention_mask).cuda()
        context_mask = (ones_mask - token_type_ids) * attention_mask
        ques_mask = token_type_ids * attention_mask
        extended_context_mask = (1.0 - context_mask) * -10000.0
        coattention_mask = torch.matmul(context_mask.unsqueeze(-1).float(), ques_mask.unsqueeze(1).float())
        # s1 = self.coatt1(sequence_output, sequence_output, mask.float())
        s1 = self.coatt1(sequence_output, sequence_output, coattention_mask)
        s1 = self.Wq1_1(torch.cat([s1, sequence_output], dim=-1)) * context_mask.unsqueeze(-1) + \
             self.Wq1_2(torch.cat([s1, sequence_output], dim=-1)) * ques_mask.unsqueeze(-1) + sequence_output
        s2 = self.coatt1(s1, s1, coattention_mask)
        start_logits = self.start_logits(s2).squeeze(-1) + extended_context_mask  # *context_mask.float()
        end_logits = self.end_logits(s2).squeeze(-1) + extended_context_mask  # *context_mask.float()

        sent_logits = self.sent(s2).squeeze(-1) * context_mask.float()
        if len(sent_logits) > 1:
            sent_logits.squeeze(-1)
        loss_fn1 = torch.nn.BCEWithLogitsLoss(reduce=False, size_average=False)
        sent_logits = sent_logits * sent_mask.float()
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sent_lbs = sent_lbs[:, 0:context_maxlen]
            # sent_weight = sent_weight[:, 0:context_maxlen]
            sent_loss = loss_fn1(sent_logits, sent_lbs.float())
            sent_loss = (sent_loss * sent_mask.float()) * sent_weight
            sent_loss = torch.sum(sent_loss, (-1, -2), keepdim=False)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)  # bsz*seq bsz*n
            end_loss = loss_fct(end_logits, end_positions)
            ans_loss = start_loss + end_loss
            total_loss = ans_loss + 0.2 * sent_loss
            return total_loss, start_logits, end_logits, sent_logits
        else:
            start_logits = nn.Softmax(dim=-1)(start_logits)
            end_logits = nn.Softmax(dim=-1)(end_logits)
            sent_logits = torch.sigmoid(sent_logits)
            return start_logits, end_logits, sent_logits


class BertForQuestionAnsweringCoAttention(BertPreTrainedModel):
    # 不expand可以吗 mask应该乘还是加
    def __init__(self, config):
        super(BertForQuestionAnsweringCoAttention, self).__init__(config)
        self.bert = BertModel(config)

        self.Wq3_1 = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.Wq3_2 = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.coatt1 = CoattentionModel(config)

        self.start_logits = nn.Linear(config.hidden_size, 1)
        self.end_logits = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sent = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self, input_ids, attention_mask, token_type_ids, start_positions=None, end_positions=None,
                sent_mask=None, sent_lbs=None, sent_weight=None, mask=None):
        if len(input_ids.shape) < 2:
            input_ids = input_ids.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            if start_positions is not None and len(start_positions.shape) < 2:
                start_positions = start_positions.unsqueeze(0)
                end_positions = end_positions.unsqueeze(0)
                sent_mask = sent_mask.unsqueeze(0)
                sent_lbs = sent_lbs.unsqueeze(0)
                sent_weight = sent_weight.unsqueeze(0)
        sequence_output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        sequence_output = self.dropout(sequence_output)
        ones_mask = torch.ones_like(attention_mask).cuda()
        context_mask = (ones_mask - token_type_ids) * attention_mask
        ques_mask = token_type_ids * attention_mask
        coattention_mask = torch.matmul(context_mask.unsqueeze(-1).float(), ques_mask.unsqueeze(1).float())
        # s1 = self.coatt1(sequence_output, sequence_output, mask.float())
        s1 = self.coatt1(sequence_output, sequence_output, coattention_mask)
        s1 = self.Wq3_1(torch.cat([s1, sequence_output], dim=-1)) * context_mask.unsqueeze(-1) + \
             self.Wq3_2(torch.cat([s1, sequence_output], dim=-1)) * ques_mask.unsqueeze(-1) + sequence_output
        s2 = self.coatt1(s1, s1, coattention_mask)
        extended_context_mask = (1.0 - context_mask) * -10000.0
        start_logits = self.start_logits(s2).squeeze(-1) + extended_context_mask  # *context_mask.float()
        end_logits = self.end_logits(s2).squeeze(-1) + extended_context_mask  # *context_mask.float()
        # start_logits=self.start_logits(co1).squeeze(-1)
        # end_logits = self.end_logits(co1).squeeze(-1)

        sent_logits = self.sent(s2).squeeze(-1) * context_mask.float()
        if len(sent_logits) > 1:
            sent_logits.squeeze(-1)
        loss_fn1 = torch.nn.BCEWithLogitsLoss(reduce=False, size_average=False)
        sent_logits = sent_logits * sent_mask.float()
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sent_lbs = sent_lbs[:, 0:context_maxlen]
            # sent_weight = sent_weight[:, 0:context_maxlen]
            sent_loss = loss_fn1(sent_logits, sent_lbs.float())
            sent_loss = (sent_loss * sent_mask.float()) * sent_weight
            sent_loss = torch.sum(sent_loss, (-1, -2), keepdim=False)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)  # bsz*seq bsz*n
            end_loss = loss_fct(end_logits, end_positions)
            ans_loss = start_loss + end_loss
            total_loss = ans_loss + 0.2 * sent_loss
            return total_loss, start_logits, end_logits, sent_logits
        else:
            # start_logits=torch.nn.functional.log_softmax(start_logits, dim=-1)
            # end_logits = torch.nn.functional.log_softmax(end_logits, dim=-1)
            start_logits = nn.Softmax(dim=-1)(start_logits)
            end_logits = nn.Softmax(dim=-1)(end_logits)
            sent_logits = torch.sigmoid(sent_logits)
            return start_logits, end_logits, sent_logits


class BertForQuestionAnsweringTwoCoAttention(BertPreTrainedModel):
    # 不expand可以吗 mask应该乘还是加
    def __init__(self, config):
        super(BertForQuestionAnsweringTwoCoAttention, self).__init__(config)
        self.bert = BertModel(config)

        self.Wq3_1 = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.Wq3_2 = nn.Linear(2 * config.hidden_size, config.hidden_size)
        # self.Wq4_1 = nn.Linear(2 * config.hidden_size, config.hidden_size)
        # self.Wq4_2 = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.Wq5_1 = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.Wq5_2 = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.coatt1 = CoattentionModel(config)
        # self.coatt2 = CoattentionModel(config)
        self.coatt3 = CoattentionModel(config)

        self.start_logits = nn.Linear(config.hidden_size, 1)
        self.end_logits = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sent = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self, input_ids, attention_mask, token_type_ids, start_positions=None, end_positions=None,
                sent_mask=None, sent_lbs=None, sent_weight=None, mask=None):
        if len(input_ids.shape) < 2:
            input_ids = input_ids.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            if start_positions is not None and len(start_positions.shape) < 2:
                start_positions = start_positions.unsqueeze(0)
                end_positions = end_positions.unsqueeze(0)
                sent_mask = sent_mask.unsqueeze(0)
                sent_lbs = sent_lbs.unsqueeze(0)
                sent_weight = sent_weight.unsqueeze(0)
        sequence_output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        sequence_output = self.dropout(sequence_output)
        ones_mask = torch.ones_like(attention_mask).cuda()
        context_mask = (ones_mask - token_type_ids) * attention_mask
        ques_mask = token_type_ids * attention_mask
        coattention_mask = torch.matmul(context_mask.unsqueeze(-1).float(), ques_mask.unsqueeze(1).float())
        # s1 = self.coatt1(sequence_output, sequence_output, mask.float())
        s1 = self.coatt1(sequence_output, sequence_output, coattention_mask)
        s1 = self.Wq3_1(torch.cat([s1, sequence_output], dim=-1)) * context_mask.unsqueeze(-1) + \
             self.Wq3_2(torch.cat([s1, sequence_output], dim=-1)) * ques_mask.unsqueeze(-1) + sequence_output
        s2 = self.coatt1(s1, s1, coattention_mask)
        # s2 = self.Wq4_1(torch.cat([s2, s1], dim=-1))*context_mask.unsqueeze(-1)+ \
        #    self.Wq4_2(torch.cat([s2, s1], dim=-1)) * ques_mask.unsqueeze(-1)+s1
        # # s3 = self.coatt2(s2, s2, coattention_mask)
        s2 = self.Wq5_1(torch.cat([s2, s1], dim=-1)) * context_mask.unsqueeze(-1) + \
             self.Wq5_2(torch.cat([s2, s1], dim=-1)) * ques_mask.unsqueeze(-1) + s1
        s4 = self.coatt3(s2, s2, coattention_mask)

        extended_context_mask = (1.0 - context_mask) * -10000.0
        start_logits = self.start_logits(s4).squeeze(-1) + extended_context_mask  # *context_mask.float()
        end_logits = self.end_logits(s4).squeeze(-1) + extended_context_mask  # *context_mask.float()
        # start_logits=self.start_logits(co1).squeeze(-1)
        # end_logits = self.end_logits(co1).squeeze(-1)

        sent_logits = self.sent(s4).squeeze(-1) * context_mask.float()
        if len(sent_logits) > 1:
            sent_logits.squeeze(-1)
        loss_fn1 = torch.nn.BCEWithLogitsLoss(reduce=False, size_average=False)
        sent_logits = sent_logits * sent_mask.float()
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sent_lbs = sent_lbs[:, 0:context_maxlen]
            # sent_weight = sent_weight[:, 0:context_maxlen]
            sent_loss = loss_fn1(sent_logits, sent_lbs.float())
            sent_loss = (sent_loss * sent_mask.float()) * sent_weight
            sent_loss = torch.sum(sent_loss, (-1, -2), keepdim=False)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)  # bsz*seq bsz*n
            end_loss = loss_fct(end_logits, end_positions)
            ans_loss = start_loss + end_loss
            total_loss = ans_loss + 0.2 * sent_loss
            return total_loss, start_logits, end_logits, sent_logits
        else:
            # start_logits=torch.nn.functional.log_softmax(start_logits, dim=-1)
            # end_logits = torch.nn.functional.log_softmax(end_logits, dim=-1)
            start_logits = nn.Softmax(dim=-1)(start_logits)
            end_logits = nn.Softmax(dim=-1)(end_logits)
            sent_logits = torch.sigmoid(sent_logits)
            return start_logits, end_logits, sent_logits


class BertForQuestionAnsweringThreeCoAttention(BertPreTrainedModel):
    # 不expand可以吗 mask应该乘还是加
    def __init__(self, config):
        super(BertForQuestionAnsweringThreeCoAttention, self).__init__(config)
        self.bert = BertModel(config)

        self.Wq3_1 = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.Wq3_2 = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.Wq4_1 = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.Wq4_2 = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.Wq5_1 = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.Wq5_2 = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.coatt1 = CoattentionModel(config)
        self.coatt2 = CoattentionModel(config)
        self.coatt3 = CoattentionModel(config)

        self.start_logits = nn.Linear(config.hidden_size, 1)
        self.end_logits = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sent = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self, input_ids, attention_mask, token_type_ids, start_positions=None, end_positions=None,
                sent_mask=None, sent_lbs=None, sent_weight=None, mask=None):
        if len(input_ids.shape) < 2:
            input_ids = input_ids.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            if start_positions is not None and len(start_positions.shape) < 2:
                start_positions = start_positions.unsqueeze(0)
                end_positions = end_positions.unsqueeze(0)
                sent_mask = sent_mask.unsqueeze(0)
                sent_lbs = sent_lbs.unsqueeze(0)
                sent_weight = sent_weight.unsqueeze(0)
        sequence_output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        sequence_output = self.dropout(sequence_output)
        ones_mask = torch.ones_like(attention_mask).cuda()
        context_mask = (ones_mask - token_type_ids) * attention_mask
        ques_mask = token_type_ids * attention_mask
        coattention_mask = torch.matmul(context_mask.unsqueeze(-1).float(), ques_mask.unsqueeze(1).float())
        # s1 = self.coatt1(sequence_output, sequence_output, mask.float())
        s1 = self.coatt1(sequence_output, sequence_output, coattention_mask)
        s1 = self.Wq3_1(torch.cat([s1, sequence_output], dim=-1)) * context_mask.unsqueeze(-1) + \
             self.Wq3_2(torch.cat([s1, sequence_output], dim=-1)) * ques_mask.unsqueeze(-1) + sequence_output
        s2 = self.coatt1(s1, s1, coattention_mask)
        s2 = self.Wq4_1(torch.cat([s2, s1], dim=-1)) * context_mask.unsqueeze(-1) + \
             self.Wq4_2(torch.cat([s2, s1], dim=-1)) * ques_mask.unsqueeze(-1) + s1
        s3 = self.coatt2(s2, s2, coattention_mask)
        s3 = self.Wq5_1(torch.cat([s3, s2], dim=-1)) * context_mask.unsqueeze(-1) + \
             self.Wq5_2(torch.cat([s3, s2], dim=-1)) * ques_mask.unsqueeze(-1) + s2
        s4 = self.coatt3(s3, s3, coattention_mask)

        extended_context_mask = (1.0 - context_mask) * -10000.0
        start_logits = self.start_logits(s4).squeeze(-1) + extended_context_mask  # *context_mask.float()
        end_logits = self.end_logits(s4).squeeze(-1) + extended_context_mask  # *context_mask.float()
        # start_logits=self.start_logits(co1).squeeze(-1)
        # end_logits = self.end_logits(co1).squeeze(-1)

        sent_logits = self.sent(s4).squeeze(-1) * context_mask.float()
        if len(sent_logits) > 1:
            sent_logits.squeeze(-1)
        loss_fn1 = torch.nn.BCEWithLogitsLoss(reduce=False, size_average=False)
        sent_logits = sent_logits * sent_mask.float()
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sent_lbs = sent_lbs[:, 0:context_maxlen]
            # sent_weight = sent_weight[:, 0:context_maxlen]
            sent_loss = loss_fn1(sent_logits, sent_lbs.float())
            sent_loss = (sent_loss * sent_mask.float()) * sent_weight
            sent_loss = torch.sum(sent_loss, (-1, -2), keepdim=False)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)  # bsz*seq bsz*n
            end_loss = loss_fct(end_logits, end_positions)
            ans_loss = start_loss + end_loss
            total_loss = ans_loss + 0.2 * sent_loss
            return total_loss, start_logits, end_logits, sent_logits
        else:
            # start_logits=torch.nn.functional.log_softmax(start_logits, dim=-1)
            # end_logits = torch.nn.functional.log_softmax(end_logits, dim=-1)
            start_logits = nn.Softmax(dim=-1)(start_logits)
            end_logits = nn.Softmax(dim=-1)(end_logits)
            sent_logits = torch.sigmoid(sent_logits)
            return start_logits, end_logits, sent_logits


class BertForQuestionAnsweringThreeSameCoAttention(BertPreTrainedModel):
    # 不expand可以吗 mask应该乘还是加
    def __init__(self, config):
        super(BertForQuestionAnsweringThreeSameCoAttention, self).__init__(config)
        self.bert = BertModel(config)

        self.Wq3_1 = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.Wq3_2 = nn.Linear(2 * config.hidden_size, config.hidden_size)
        # self.Wq4_1 = nn.Linear(2 * config.hidden_size, config.hidden_size)
        # self.Wq4_2 = nn.Linear(2 * config.hidden_size, config.hidden_size)
        # self.Wq5_1 = nn.Linear(2 * config.hidden_size, config.hidden_size)
        # self.Wq5_2 = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.coatt1 = CoattentionModel(config)
        # self.coatt2 = CoattentionModel(config)
        # self.coatt3 = CoattentionModel(config)

        self.start_logits = nn.Linear(config.hidden_size, 1)
        self.end_logits = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sent = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self, input_ids, attention_mask, token_type_ids, start_positions=None, end_positions=None,
                sent_mask=None, sent_lbs=None, sent_weight=None, mask=None):
        if len(input_ids.shape) < 2:
            input_ids = input_ids.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            if start_positions is not None and len(start_positions.shape) < 2:
                start_positions = start_positions.unsqueeze(0)
                end_positions = end_positions.unsqueeze(0)
                sent_mask = sent_mask.unsqueeze(0)
                sent_lbs = sent_lbs.unsqueeze(0)
                sent_weight = sent_weight.unsqueeze(0)
        sequence_output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        sequence_output = self.dropout(sequence_output)
        ones_mask = torch.ones_like(attention_mask).cuda()
        context_mask = (ones_mask - token_type_ids) * attention_mask
        ques_mask = token_type_ids * attention_mask
        coattention_mask = torch.matmul(context_mask.unsqueeze(-1).float(), ques_mask.unsqueeze(1).float())
        # s1 = self.coatt1(sequence_output, sequence_output, mask.float())
        s1 = self.coatt1(sequence_output, sequence_output, coattention_mask)
        s1 = self.Wq3_1(torch.cat([s1, sequence_output], dim=-1)) * context_mask.unsqueeze(-1) + \
             self.Wq3_2(torch.cat([s1, sequence_output], dim=-1)) * ques_mask.unsqueeze(-1) + sequence_output
        s2 = self.coatt1(s1, s1, coattention_mask)
        s2 = self.Wq3_1(torch.cat([s2, s1], dim=-1)) * context_mask.unsqueeze(-1) + \
             self.Wq3_2(torch.cat([s2, s1], dim=-1)) * ques_mask.unsqueeze(-1) + s1
        s3 = self.coatt1(s2, s2, coattention_mask)
        s3 = self.Wq3_1(torch.cat([s3, s2], dim=-1)) * context_mask.unsqueeze(-1) + \
             self.Wq3_2(torch.cat([s3, s2], dim=-1)) * ques_mask.unsqueeze(-1) + s2
        s4 = self.coatt1(s3, s3, coattention_mask)

        extended_context_mask = (1.0 - context_mask) * -10000.0
        start_logits = self.start_logits(s4).squeeze(-1) + extended_context_mask  # *context_mask.float()
        end_logits = self.end_logits(s4).squeeze(-1) + extended_context_mask  # *context_mask.float()
        # start_logits=self.start_logits(co1).squeeze(-1)
        # end_logits = self.end_logits(co1).squeeze(-1)

        sent_logits = self.sent(s4).squeeze(-1) * context_mask.float()
        if len(sent_logits) > 1:
            sent_logits.squeeze(-1)
        loss_fn1 = torch.nn.BCEWithLogitsLoss(reduce=False, size_average=False)
        sent_logits = sent_logits * sent_mask.float()
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sent_lbs = sent_lbs[:, 0:context_maxlen]
            # sent_weight = sent_weight[:, 0:context_maxlen]
            sent_loss = loss_fn1(sent_logits, sent_lbs.float())
            sent_loss = (sent_loss * sent_mask.float()) * sent_weight
            sent_loss = torch.sum(sent_loss, (-1, -2), keepdim=False)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)  # bsz*seq bsz*n
            end_loss = loss_fct(end_logits, end_positions)
            ans_loss = start_loss + end_loss
            total_loss = ans_loss + 0.2 * sent_loss
            return total_loss, start_logits, end_logits, sent_logits
        else:
            # start_logits=torch.nn.functional.log_softmax(start_logits, dim=-1)
            # end_logits = torch.nn.functional.log_softmax(end_logits, dim=-1)
            start_logits = nn.Softmax(dim=-1)(start_logits)
            end_logits = nn.Softmax(dim=-1)(end_logits)
            sent_logits = torch.sigmoid(sent_logits)
            return start_logits, end_logits, sent_logits


class BertForQuestionAnsweringGraph(BertPreTrainedModel):
    # 不expand可以吗 mask应该乘还是加
    def __init__(self, config):
        super(BertForQuestionAnsweringGraph, self).__init__(config)
        self.bert = BertModel(config)

        self.Wq3_1 = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.Wq3_2 = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.coatt1 = CoattentionModel(config)

        self.start_logits = nn.Linear(config.hidden_size, 1)
        self.end_logits = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sent = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self, input_ids, attention_mask, token_type_ids, start_positions=None, end_positions=None,
                sent_mask=None, sent_lbs=None, sent_weight=None, mask=None):
        if len(input_ids.shape) < 2:
            input_ids = input_ids.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            if start_positions is not None and len(start_positions.shape) < 2:
                start_positions = start_positions.unsqueeze(0)
                end_positions = end_positions.unsqueeze(0)
                sent_mask = sent_mask.unsqueeze(0)
                sent_lbs = sent_lbs.unsqueeze(0)
                sent_weight = sent_weight.unsqueeze(0)
        sequence_output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        sequence_output = self.dropout(sequence_output)
        ones_mask = torch.ones_like(attention_mask).cuda()
        context_mask = (ones_mask - token_type_ids) * attention_mask
        ques_mask = token_type_ids * attention_mask
        coattention_mask = torch.matmul(context_mask.unsqueeze(-1).float(), ques_mask.unsqueeze(1).float())
        s1 = self.coatt1(sequence_output, sequence_output, mask.float())
        s1 = self.Wq3_1(torch.cat([s1, sequence_output], dim=-1)) * context_mask.unsqueeze(-1) + \
             self.Wq3_2(torch.cat([s1, sequence_output], dim=-1)) * ques_mask.unsqueeze(-1) + sequence_output
        s2 = self.coatt1(s1, s1, coattention_mask)
        extended_context_mask = (1.0 - context_mask) * -10000.0
        start_logits = self.start_logits(s2).squeeze(-1) + extended_context_mask  # *context_mask.float()
        end_logits = self.end_logits(s2).squeeze(-1) + extended_context_mask  # *context_mask.float()

        sent_logits = self.sent(s2).squeeze(-1) * context_mask.float()
        if len(sent_logits) > 1:
            sent_logits.squeeze(-1)
        loss_fn1 = torch.nn.BCEWithLogitsLoss(reduce=False, size_average=False)
        sent_logits = sent_logits * sent_mask.float()
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sent_lbs = sent_lbs[:, 0:context_maxlen]
            # sent_weight = sent_weight[:, 0:context_maxlen]
            sent_loss = loss_fn1(sent_logits, sent_lbs.float())
            sent_loss = (sent_loss * sent_mask.float()) * sent_weight
            sent_loss = torch.sum(sent_loss, (-1, -2), keepdim=False)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)  # bsz*seq bsz*n
            end_loss = loss_fct(end_logits, end_positions)
            ans_loss = start_loss + end_loss
            total_loss = ans_loss + 0.2 * sent_loss
            return total_loss, start_logits, end_logits, sent_logits
        else:
            # start_logits=torch.nn.functional.log_softmax(start_logits, dim=-1)
            # end_logits = torch.nn.functional.log_softmax(end_logits, dim=-1)
            start_logits = nn.Softmax(dim=-1)(start_logits)
            end_logits = nn.Softmax(dim=-1)(end_logits)
            sent_logits = torch.sigmoid(sent_logits)
            return start_logits, end_logits, sent_logits
