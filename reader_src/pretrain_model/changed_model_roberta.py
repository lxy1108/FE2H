import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import math

from transformers import ElectraModel, AlbertModel, BertModel, RobertaModel, DebertaModel
from transformer import TransformerLayer
from qanet import MyQANet
from coattention import CoattentionModel
from biattention import BiAttention


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


class ElectraForParagraphClassification(ElectraModel):
    def __init__(self, config):
        super(ElectraForParagraphClassification, self).__init__(config)
        self.bert = ElectraModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, cls_mask=None, cls_label=None, cls_weight=None):
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            if cls_label is not None:
                cls_mask = cls_mask.unsqueeze(0)
                cls_label = cls_label.unsqueeze(0)
                cls_weight = cls_weight.unsqueeze(0)
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        cls_output = outputs[0]
        cls_output = torch.index_select(cls_output, dim=1, index=torch.tensor([0, ]).cuda())
        cls_output = cls_output.squeeze(dim=1)
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output).squeeze(-1)
        if cls_label is None:
            return logits
        # 选择第一个标记结果
        cls_label = torch.index_select(cls_label, dim=1, index=torch.tensor([0, ]).cuda())
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.config.num_labels), cls_label.view(-1))
        return loss, logits


class ElectraForRelatedSentence(ElectraModel):

    def __init__(self, config):
        super(ElectraForRelatedSentence, self).__init__(config)
        self.robert = ElectraModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, cls_mask=None, cls_label=None, cls_weight=None):
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            if cls_label is not None:
                cls_mask = cls_mask.unsqueeze(0)
                cls_label = cls_label.unsqueeze(0)
                cls_weight = cls_weight.unsqueeze(0)
        sequence_output = self.robert(input_ids=input_ids,
                                      attention_mask=attention_mask,
                                      token_type_ids=token_type_ids)
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


def split_context_ques(sequence_output, pq_end_pos):
    context_max_len = 512 - 64
    ques_max_len = 64
    sep_tok_len = 1
    context_sequence_output = sequence_output.new(
        torch.Size((sequence_output.size(0), context_max_len, sequence_output.size(2)))).zero_()
    ques_sequence_output = sequence_output.new_zeros(
        (sequence_output.size(0), ques_max_len, sequence_output.size(2)))
    ques_attention_mask = sequence_output.new_zeros((sequence_output.size(0), ques_max_len))
    context_attention_mask = sequence_output.new_zeros((sequence_output.size(0), context_max_len))
    for i in range(0, sequence_output.size(0)):
        q_end = pq_end_pos[i][0]
        p_end = pq_end_pos[i][1]
        context_sequence_output[i, :min(context_max_len, q_end)] = sequence_output[i, 1: 1 + min(context_max_len, q_end)]
        ques_sequence_output[i, :min(ques_max_len, p_end - q_end - sep_tok_len)] = sequence_output[i, q_end + sep_tok_len + 1: q_end + sep_tok_len + 1 + min(p_end - q_end - sep_tok_len, ques_max_len)]
        ques_attention_mask[i, :min(ques_max_len, p_end - q_end - sep_tok_len)] = sequence_output.new_ones(
            (1, ques_max_len))[0, :min(ques_max_len, p_end - q_end - sep_tok_len)]
        context_attention_mask[i, : min(context_max_len, q_end)] = sequence_output.new_ones((1, context_max_len))[0,
                                                                   : min(context_max_len, q_end)]
    return context_sequence_output, ques_sequence_output, context_attention_mask, ques_attention_mask


def masked_softmax(vector: torch.Tensor,
                   mask: torch.Tensor,
                   dim: int = -1,
                   memory_efficient: bool = False,
                   mask_fill_value: float = -1e32) -> torch.Tensor:
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        #mask = mask.half()

        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the mask, we zero these out.
            result = torch.nn.functional.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_vector = vector.masked_fill((1 - mask).byte(), mask_fill_value)
            result = torch.nn.functional.softmax(masked_vector, dim=dim)
    return result


class MatchAttention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MatchAttention, self).__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(input_size, hidden_size)
        self.map_linear = nn.Linear(hidden_size, hidden_size)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.W.weight.data)
        self.W.bias.data.fill_(0.1)

    def forward(self, passage, question, q_mask):
        Wp = passage
        Wq = question
        scores = torch.bmm(Wp, Wq.transpose(2, 1))
        mask = q_mask.unsqueeze(1).repeat(1, passage.size(1), 1)
        # scores.data.masked_fill_(mask.data, -float('inf'))
        alpha = masked_softmax(scores, mask)
        output = torch.bmm(alpha, Wq)
        output = nn.ReLU()(self.map_linear(output))
        #output = self.map_linear(all_con)
        return output


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
        w = self.dense.weight.t().view(self.num_attention_heads, self.attention_head_size, self.hidden_size).to(context_layer.dtype)
        b = self.dense.bias.to(context_layer.dtype)

        projected_context_layer = torch.einsum("bfnd,ndh->bfh", context_layer, w) + b
        projected_context_layer_dropout = self.dropout(projected_context_layer)
        layernormed_context_layer = self.LayerNorm(input_ids_1 + projected_context_layer_dropout)

        ffn_output = self.ffn(layernormed_context_layer)
        ffn_output = self.activation(ffn_output)
        ffn_output = self.ffn_output(ffn_output)
        hidden_states = self.full_layer_layer_norm(ffn_output + layernormed_context_layer)
        return hidden_states


class AlbertForQuestionAnsweringForwardBestWithMask(AlbertModel):
    def __init__(self, config):
        super(AlbertForQuestionAnsweringForwardBestWithMask, self).__init__(config)
        self.albert = AlbertModel(config)
        self.start_logits = nn.Linear(config.hidden_size, 1)
        self.end_logits = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sent = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                cls_weight=None,
                entity_ids=None,
                pq_end_pos=None,
                start_positions=None,
                end_positions=None,
                sent_mask=None,
                sent_lbs=None,
                sent_weight=None):
        if len(input_ids.shape) < 2:
            input_ids = input_ids.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            pq_end_pos = pq_end_pos.unsqueeze(0)
            if start_positions is not None and len(start_positions.shape) < 2:
                start_positions = start_positions.unsqueeze(0)
                end_positions = end_positions.unsqueeze(0)
                sent_mask = sent_mask.unsqueeze(0)
                sent_lbs = sent_lbs.unsqueeze(0)
                sent_weight = sent_weight.unsqueeze(0)
        # roberta 取消了NSP任务，所以token_type_ids没有作用
        sequence_output = self.albert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = sequence_output[0]
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
            # sent_loss = (sent_loss * sent_mask.float()) * sent_weight
            sent_loss = sent_loss * sent_mask.float()
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


class ElectraForQuestionAnsweringForwardBest(ElectraModel):
    def __init__(self, config):
        super(ElectraForQuestionAnsweringForwardBest, self).__init__(config)
        self.electra = ElectraModel(config)
        self.start_logits = nn.Linear(config.hidden_size, 1)
        self.end_logits = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sent = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                cls_weight=None,
                entity_ids=None,
                pq_end_pos=None,
                start_positions=None,
                end_positions=None,
                sent_mask=None,
                sent_lbs=None,
                sent_weight=None):
        if len(input_ids.shape) < 2:
            input_ids = input_ids.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            pq_end_pos = pq_end_pos.unsqueeze(0)
            if start_positions is not None and len(start_positions.shape) < 2:
                start_positions = start_positions.unsqueeze(0)
                end_positions = end_positions.unsqueeze(0)
                sent_mask = sent_mask.unsqueeze(0)
                sent_lbs = sent_lbs.unsqueeze(0)
                sent_weight = sent_weight.unsqueeze(0)
        # roberta 取消了NSP任务，所以token_type_ids没有作用
        sequence_output = self.electra(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = sequence_output[0]
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
        # sent_logits = sent_logits * sent_mask.float()
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sent_lbs = sent_lbs[:, 0:context_maxlen]
            # sent_weight = sent_weight[:, 0:context_maxlen]
            sent_loss = loss_fn1(sent_logits, sent_lbs.float())
            # sent_loss = (sent_loss * sent_mask.float()) * sent_weight
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


class AlbertForQuestionAnsweringForwardBest(AlbertModel):
    def __init__(self, config):
        super(AlbertForQuestionAnsweringForwardBest, self).__init__(config)
        self.albert = AlbertModel(config)
        self.start_logits = nn.Linear(config.hidden_size, 1)
        self.end_logits = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sent = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                cls_weight=None,
                entity_ids=None,
                pq_end_pos=None,
                start_positions=None,
                end_positions=None,
                sent_mask=None,
                sent_lbs=None,
                sent_weight=None):
        if len(input_ids.shape) < 2:
            input_ids = input_ids.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            pq_end_pos = pq_end_pos.unsqueeze(0)
            if start_positions is not None and len(start_positions.shape) < 2:
                start_positions = start_positions.unsqueeze(0)
                end_positions = end_positions.unsqueeze(0)
                sent_mask = sent_mask.unsqueeze(0)
                sent_lbs = sent_lbs.unsqueeze(0)
                sent_weight = sent_weight.unsqueeze(0)
        # roberta 取消了NSP任务，所以token_type_ids没有作用
        sequence_output = self.albert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = sequence_output[0]
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
        # sent_logits = sent_logits * sent_mask.float()
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sent_lbs = sent_lbs[:, 0:context_maxlen]
            # sent_weight = sent_weight[:, 0:context_maxlen]
            sent_loss = loss_fn1(sent_logits, sent_lbs.float())
            # sent_loss = (sent_loss * sent_mask.float()) * sent_weight
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


class ElectraForQuestionAnsweringForwardBestWithNoise(ElectraModel):
    def __init__(self, config):
        super(ElectraForQuestionAnsweringForwardBestWithNoise, self).__init__(config)
        self.electra = ElectraModel(config)
        self.start_logits = nn.Linear(config.hidden_size, 1)
        self.end_logits = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sent = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                cls_weight=None,
                entity_ids=None,
                pq_end_pos=None,
                start_positions=None,
                end_positions=None,
                sent_mask=None,
                sent_lbs=None,
                sent_weight=None):
        if len(input_ids.shape) < 2:
            input_ids = input_ids.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            pq_end_pos = pq_end_pos.unsqueeze(0)
            if start_positions is not None and len(start_positions.shape) < 2:
                start_positions = start_positions.unsqueeze(0)
                end_positions = end_positions.unsqueeze(0)
                sent_mask = sent_mask.unsqueeze(0)
                sent_lbs = sent_lbs.unsqueeze(0)
                sent_weight = sent_weight.unsqueeze(0)
        # roberta 取消了NSP任务，所以token_type_ids没有作用
        sequence_output = self.electra(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = sequence_output[0]
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
        # sent_logits = sent_logits * sent_mask.float()
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sent_lbs = sent_lbs[:, 0:context_maxlen]
            # sent_weight = sent_weight[:, 0:context_maxlen]
            sent_loss = loss_fn1(sent_logits, sent_lbs.float())
            # sent_loss = (sent_loss * sent_mask.float()) * sent_weight
            # sent_loss = (sent_loss * sent_mask.float())
            sent_loss = torch.sum(sent_loss, (-1, -2), keepdim=False)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            # 对start pos 做noise操作
            l_start_positions = start_positions - 1
            r_start_positions = start_positions + 1
            l_start_positions[l_start_positions <= 1] = l_start_positions[l_start_positions <= 1] + 1
            r_start_positions[r_start_positions <= 3] = r_start_positions[r_start_positions <= 3] - 1
            r_start_positions[r_start_positions >= 512] = r_start_positions[r_start_positions >= 512] - 1
            # 对end pos 做noise操作
            l_end_positions = end_positions - 1
            r_end_positions = end_positions + 1
            l_end_positions[l_end_positions <= 1] = l_end_positions[l_end_positions <= 1] + 1
            r_end_positions[r_end_positions <= 3] = r_end_positions[r_end_positions <= 3] - 1
            r_end_positions[r_end_positions >= 512] = r_end_positions[r_end_positions >= 512] - 1

            # start loss 加和
            start_loss = loss_fct(start_logits, start_positions)
            l_start_loss = loss_fct(start_logits, l_start_positions)
            r_start_loss = loss_fct(start_logits, r_start_positions)
            start_loss = 0.5 * start_loss + 0.25 * l_start_loss + 0.25 * r_start_loss
            # end loss 加和
            end_loss = loss_fct(end_logits, end_positions)
            l_end_loss = loss_fct(end_logits, l_end_positions)
            r_end_loss = loss_fct(end_logits, r_end_positions)
            end_loss = 0.5 * end_loss + 0.25 * l_end_loss + 0.25 * r_end_loss

            ans_loss = start_loss + end_loss
            total_loss = ans_loss + 0.2 * sent_loss
            return total_loss, start_logits, end_logits, sent_logits
        else:
            start_logits = nn.Softmax(dim=-1)(start_logits)
            end_logits = nn.Softmax(dim=-1)(end_logits)
            sent_logits = torch.sigmoid(sent_logits)
            return start_logits, end_logits, sent_logits



class ElectraForQuestionAnsweringMatchAttention(ElectraModel):
    def __init__(self, config):
        super(ElectraForQuestionAnsweringMatchAttention, self).__init__(config)
        self.electra = ElectraModel(config)
        self.match_attention = MatchAttention(input_size=config.hidden_size, hidden_size=config.hidden_size)
        self.start_logits = nn.Linear(config.hidden_size, 1)
        self.end_logits = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sent = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                cls_weight=None,
                entity_ids=None,
                pq_end_pos=None,
                start_positions=None,
                end_positions=None,
                sent_mask=None,
                sent_lbs=None,
                sent_weight=None):
        if len(input_ids.shape) < 2:
            input_ids = input_ids.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            pq_end_pos = pq_end_pos.unsqueeze(0)
            if start_positions is not None and len(start_positions.shape) < 2:
                start_positions = start_positions.unsqueeze(0)
                end_positions = end_positions.unsqueeze(0)
                sent_mask = sent_mask.unsqueeze(0)
                sent_lbs = sent_lbs.unsqueeze(0)
                sent_weight = sent_weight.unsqueeze(0)
        # roberta 取消了NSP任务，所以token_type_ids没有作用
        outputs = self.electra(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        ## add match attention
        context_sequence_output, ques_sequence_output, context_attention_mask, ques_attention_mask = split_context_ques(
            sequence_output, pq_end_pos)
        sequence_output = self.match_attention(sequence_output, ques_sequence_output, ques_attention_mask)
        sequence_output = sequence_output + outputs[0]
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
        # sent_logits = sent_logits * sent_mask.float()
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sent_lbs = sent_lbs[:, 0:context_maxlen]
            # sent_weight = sent_weight[:, 0:context_maxlen]
            sent_loss = loss_fn1(sent_logits, sent_lbs.float())
            # sent_loss = (sent_loss * sent_mask.float()) * sent_weight
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


class AlbertForQuestionAnsweringCrossAttention(AlbertModel):
    def __init__(self, config):
        super(AlbertForQuestionAnsweringCrossAttention, self).__init__(config)
        self.albert = AlbertModel(config)
        self.cross_attention = CrossAttention(config=config)
        self.start_logits = nn.Linear(config.hidden_size, 1)
        self.end_logits = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sent = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                cls_weight=None,
                entity_ids=None,
                pq_end_pos=None,
                start_positions=None,
                end_positions=None,
                sent_mask=None,
                sent_lbs=None,
                sent_weight=None):
        if len(input_ids.shape) < 2:
            input_ids = input_ids.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            pq_end_pos = pq_end_pos.unsqueeze(0)
            if start_positions is not None and len(start_positions.shape) < 2:
                start_positions = start_positions.unsqueeze(0)
                end_positions = end_positions.unsqueeze(0)
                sent_mask = sent_mask.unsqueeze(0)
                sent_lbs = sent_lbs.unsqueeze(0)
                sent_weight = sent_weight.unsqueeze(0)
        # roberta 取消了NSP任务，所以token_type_ids没有作用
        outputs = self.albert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        ## add cross attention
        context_sequence_output, ques_sequence_output, context_attention_mask, ques_attention_mask = split_context_ques(
            sequence_output, pq_end_pos)
        sequence_output = self.cross_attention(ques_sequence_output, sequence_output, ques_attention_mask)
        sequence_output = sequence_output + outputs[0]
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
        # sent_logits = sent_logits * sent_mask.float()
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sent_lbs = sent_lbs[:, 0:context_maxlen]
            # sent_weight = sent_weight[:, 0:context_maxlen]
            sent_loss = loss_fn1(sent_logits, sent_lbs.float())
            # sent_loss = (sent_loss * sent_mask.float()) * sent_weight
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


class ElectraForQuestionAnsweringCrossAttention(ElectraModel):
    def __init__(self, config):
        super(ElectraForQuestionAnsweringCrossAttention, self).__init__(config)
        self.electra = ElectraModel(config)
        self.cross_attention = CrossAttention(config=config)
        self.start_logits = nn.Linear(config.hidden_size, 1)
        self.end_logits = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sent = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                cls_weight=None,
                entity_ids=None,
                pq_end_pos=None,
                start_positions=None,
                end_positions=None,
                sent_mask=None,
                sent_lbs=None,
                sent_weight=None):
        if len(input_ids.shape) < 2:
            input_ids = input_ids.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            pq_end_pos = pq_end_pos.unsqueeze(0)
            if start_positions is not None and len(start_positions.shape) < 2:
                start_positions = start_positions.unsqueeze(0)
                end_positions = end_positions.unsqueeze(0)
                sent_mask = sent_mask.unsqueeze(0)
                sent_lbs = sent_lbs.unsqueeze(0)
                sent_weight = sent_weight.unsqueeze(0)
        # roberta 取消了NSP任务，所以token_type_ids没有作用
        outputs = self.electra(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        # add cross attention
        context_sequence_output, ques_sequence_output, context_attention_mask, ques_attention_mask = split_context_ques(
            sequence_output, pq_end_pos)
        sequence_output = self.cross_attention(ques_sequence_output, sequence_output, ques_attention_mask)
        sequence_output = sequence_output + outputs[0]
        ones_mask = torch.ones_like(attention_mask).cuda()
        context_mask = (ones_mask - token_type_ids) * attention_mask
        extended_context_mask = (1.0 - context_mask) * -10000.0
        start_logits = self.start_logits(sequence_output).squeeze(-1) + extended_context_mask
        end_logits = self.end_logits(sequence_output).squeeze(-1) + extended_context_mask
        # 去除context mask
        sent_logits = self.sent(sequence_output).squeeze(-1) * context_mask.float()
        # sent_logits = self.sent(sequence_output).squeeze(-1)
        if len(sent_logits) > 1:
            sent_logits.squeeze(-1)
        loss_fn1 = torch.nn.BCEWithLogitsLoss(reduce=False, size_average=False)
        # 去除sent_mask
        # sent_logits = sent_logits * sent_mask.float()
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sent_lbs = sent_lbs[:, 0:context_maxlen]
            # sent_weight = sent_weight[:, 0:context_maxlen]
            sent_loss = loss_fn1(sent_logits, sent_lbs.float())
            # sent_loss = (sent_loss * sent_mask.float()) * sent_weight
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


class BertForQuestionAnsweringQANet(BertModel):
    def __init__(self, config):
        super(BertForQuestionAnsweringQANet, self).__init__(config)
        self.bert = BertModel(config)
        self.coattention = CoattentionModel(config=config)
        self.cross_attention = CrossAttention(config=config)
        self.true_coattention = MyQANet(d_model=config.hidden_size)
        # self.bi_attention = BiAttention(config.hidden_size, config.hidden_size, config.hidden_size)
        # self.bi_attn_linear = nn.Linear(config.hidden_size * 4, config.hidden_size)
        # self.bi_attn_dropout = nn.Dropout(0.2)
        # self.w1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # self.w2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # self.w3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # initialization
        # self.w1.data.fill_(0.5)
        # self.w2.data.fill_(0.5)
        # self.w3.data.fill_(0.3)
        self.start_logits = nn.Linear(config.hidden_size, 1)
        self.end_logits = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sent = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                cls_weight=None,
                entity_ids=None,
                pq_end_pos=None,
                start_positions=None,
                end_positions=None,
                sent_mask=None,
                sent_lbs=None,
                sent_weight=None):
        if len(input_ids.shape) < 2:
            input_ids = input_ids.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            pq_end_pos = pq_end_pos.unsqueeze(0)
            if start_positions is not None and len(start_positions.shape) < 2:
                start_positions = start_positions.unsqueeze(0)
                end_positions = end_positions.unsqueeze(0)
                sent_mask = sent_mask.unsqueeze(0)
                sent_lbs = sent_lbs.unsqueeze(0)
                sent_weight = sent_weight.unsqueeze(0)
        # roberta 取消了NSP任务，所以token_type_ids没有作用
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        # add cross attention
        context_sequence_output, ques_sequence_output, context_attention_mask, ques_attention_mask = split_context_ques(
            sequence_output, pq_end_pos)
        # cross_output = self.cross_attention(ques_sequence_output, sequence_output, ques_attention_mask)
        true_coattention = self.true_coattention(C=context_sequence_output, Q=ques_sequence_output, cmask=context_attention_mask, qmask=ques_attention_mask)
        bs, context_len, hidden_dim = true_coattention.shape
        zeros = torch.zeros((bs, 512 - context_len, hidden_dim)).cuda()
        true_coattention = torch.cat([true_coattention, zeros], dim=1)
        # sequence_output = self.qa_net(C=context_sequence_output,
        #                               Q=ques_sequence_output,
        #                               cmask=context_attention_mask,
        #                               qmask=ques_attention_mask)
        # ones_mask = torch.ones_like(attention_mask).cuda()
        # context_mask = (ones_mask - token_type_ids) * attention_mask
        # ques_mask = token_type_ids * attention_mask
        # coattention_mask = torch.matmul(context_mask.unsqueeze(-1).float(), ques_mask.unsqueeze(1).float())
        # sequence_output = self.coattention(sequence_output, sequence_output, coattention_mask)

        # biattention
        # bi_attention_output, memory = self.bi_attention(context_sequence_output, ques_sequence_output,
        #                                                 ques_attention_mask)
        # bi_attention_output = self.bi_attn_linear(bi_attention_output)
        # bs, context_len, hidden_dim = bi_attention_output.shape
        # zeros = torch.zeros((bs, 512 - context_len, hidden_dim)).cuda()
        # bi_attention_output = torch.cat([bi_attention_output, zeros], dim=1)

        # bs, context_len, hidden_dim = sequence_output.shape
        # zeros = torch.zeros((bs, 512 - context_len, hidden_dim)).cuda()
        # sequence_output = torch.cat([sequence_output, zeros], dim=1)
        # sequence_output = self.w1 * sequence_output + self.w2 * cross_output + outputs[0] + self.w3 * bi_attention_output
        sequence_output = true_coattention
        # sequence_output = 0.3 * sequence_output + 0.7 * cross_output
        # sequence_output = 0.5 * sequence_output + 0.5 * cross_output + outputs[0]
        sequence_output = sequence_output + outputs[0]
        ones_mask = torch.ones_like(attention_mask).cuda()
        context_mask = (ones_mask - token_type_ids) * attention_mask
        extended_context_mask = (1.0 - context_mask) * -10000.0
        start_logits = self.start_logits(sequence_output).squeeze(-1) + extended_context_mask
        end_logits = self.end_logits(sequence_output).squeeze(-1) + extended_context_mask
        # 去除context mask
        sent_logits = self.sent(sequence_output).squeeze(-1) * context_mask.float()
        # sent_logits = self.sent(sequence_output).squeeze(-1)
        if len(sent_logits) > 1:
            sent_logits.squeeze(-1)
        loss_fn1 = torch.nn.BCEWithLogitsLoss(reduce=False, size_average=False)
        # 去除sent_mask
        # sent_logits = sent_logits * sent_mask.float()
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sent_lbs = sent_lbs[:, 0:context_maxlen]
            # sent_weight = sent_weight[:, 0:context_maxlen]
            sent_loss = loss_fn1(sent_logits, sent_lbs.float())
            # sent_loss = (sent_loss * sent_mask.float()) * sent_weight
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


class ElectraForQuestionAnsweringQANetTrueCoAttention(ElectraModel):
    def __init__(self, config):
        super(ElectraForQuestionAnsweringQANetTrueCoAttention, self).__init__(config)
        self.electra = ElectraModel(config)
        self.cross_attention = CrossAttention(config=config)
        self.fake_cross_attention = CoattentionModel(config=config)
        self.true_coattention = MyQANet(d_model=config.hidden_size)
        self.w1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # initialization
        self.w1.data.fill_(0.5)
        self.w2.data.fill_(0.5)
        self.w3.data.fill_(0.3)
        self.start_logits = nn.Linear(config.hidden_size, 1)
        self.end_logits = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sent = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                cls_weight=None,
                entity_ids=None,
                pq_end_pos=None,
                start_positions=None,
                end_positions=None,
                sent_mask=None,
                sent_lbs=None,
                sent_weight=None):
        if len(input_ids.shape) < 2:
            input_ids = input_ids.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            pq_end_pos = pq_end_pos.unsqueeze(0)
            if start_positions is not None and len(start_positions.shape) < 2:
                start_positions = start_positions.unsqueeze(0)
                end_positions = end_positions.unsqueeze(0)
                sent_mask = sent_mask.unsqueeze(0)
                sent_lbs = sent_lbs.unsqueeze(0)
                sent_weight = sent_weight.unsqueeze(0)
        # roberta 取消了NSP任务，所以token_type_ids没有作用
        outputs = self.electra(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        # add cross attention
        context_sequence_output, ques_sequence_output, context_attention_mask, ques_attention_mask = split_context_ques(
            sequence_output, pq_end_pos)
        cross_output = self.cross_attention(ques_sequence_output, sequence_output, ques_attention_mask)
        true_coattention = self.true_coattention(C=context_sequence_output, Q=ques_sequence_output, cmask=context_attention_mask, qmask=ques_attention_mask)
        bs, context_len, hidden_dim = true_coattention.shape
        zeros = torch.zeros((bs, 512 - context_len, hidden_dim)).cuda()
        true_coattention = torch.cat([true_coattention, zeros], dim=1)

        ones_mask = torch.ones_like(attention_mask).cuda()
        context_mask = (ones_mask - token_type_ids) * attention_mask
        ques_mask = token_type_ids * attention_mask
        coattention_mask = torch.matmul(context_mask.unsqueeze(-1).float(), ques_mask.unsqueeze(1).float())
        fake_cross_attention = self.fake_cross_attention(sequence_output, sequence_output, coattention_mask)

        sequence_output = cross_output * self.w1 + true_coattention * self.w2 + fake_cross_attention * self.w3
        sequence_output = sequence_output + outputs[0]
        ones_mask = torch.ones_like(attention_mask).cuda()
        context_mask = (ones_mask - token_type_ids) * attention_mask
        extended_context_mask = (1.0 - context_mask) * -10000.0
        start_logits = self.start_logits(sequence_output).squeeze(-1) + extended_context_mask
        end_logits = self.end_logits(sequence_output).squeeze(-1) + extended_context_mask
        # 去除context mask
        sent_logits = self.sent(sequence_output).squeeze(-1) * context_mask.float()
        # sent_logits = self.sent(sequence_output).squeeze(-1)
        if len(sent_logits) > 1:
            sent_logits.squeeze(-1)
        loss_fn1 = torch.nn.BCEWithLogitsLoss(reduce=False, size_average=False)
        # 去除sent_mask
        # sent_logits = sent_logits * sent_mask.float()
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sent_lbs = sent_lbs[:, 0:context_maxlen]
            # sent_weight = sent_weight[:, 0:context_maxlen]
            sent_loss = loss_fn1(sent_logits, sent_lbs.float())
            # sent_loss = (sent_loss * sent_mask.float()) * sent_weight
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


class BertForQuestionAnsweringQANetTrueCoAttention(BertModel):
    def __init__(self, config):
        super(BertForQuestionAnsweringQANetTrueCoAttention, self).__init__(config)
        self.bert = BertModel(config)
        self.cross_attention = CrossAttention(config=config)
        self.true_coattention = MyQANet(d_model=config.hidden_size)
        self.start_logits = nn.Linear(config.hidden_size, 1)
        self.end_logits = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sent = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                cls_weight=None,
                entity_ids=None,
                pq_end_pos=None,
                start_positions=None,
                end_positions=None,
                sent_mask=None,
                sent_lbs=None,
                sent_weight=None):
        if len(input_ids.shape) < 2:
            input_ids = input_ids.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            pq_end_pos = pq_end_pos.unsqueeze(0)
            if start_positions is not None and len(start_positions.shape) < 2:
                start_positions = start_positions.unsqueeze(0)
                end_positions = end_positions.unsqueeze(0)
                sent_mask = sent_mask.unsqueeze(0)
                sent_lbs = sent_lbs.unsqueeze(0)
                sent_weight = sent_weight.unsqueeze(0)
        # roberta 取消了NSP任务，所以token_type_ids没有作用
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        # add cross attention
        context_sequence_output, ques_sequence_output, context_attention_mask, ques_attention_mask = split_context_ques(
            sequence_output, pq_end_pos)
        cross_output = self.cross_attention(ques_sequence_output, sequence_output, ques_attention_mask)
        true_coattention = self.true_coattention(C=context_sequence_output, Q=ques_sequence_output, cmask=context_attention_mask, qmask=ques_attention_mask)
        bs, context_len, hidden_dim = true_coattention.shape
        zeros = torch.zeros((bs, 512 - context_len, hidden_dim)).cuda()
        true_coattention = torch.cat([true_coattention, zeros], dim=1)
        sequence_output = cross_output * 0.5 + true_coattention * 0.5
        sequence_output = sequence_output + outputs[0]
        ones_mask = torch.ones_like(attention_mask).cuda()
        context_mask = (ones_mask - token_type_ids) * attention_mask
        extended_context_mask = (1.0 - context_mask) * -10000.0
        start_logits = self.start_logits(sequence_output).squeeze(-1) + extended_context_mask
        end_logits = self.end_logits(sequence_output).squeeze(-1) + extended_context_mask
        # 去除context mask
        sent_logits = self.sent(sequence_output).squeeze(-1) * context_mask.float()
        # sent_logits = self.sent(sequence_output).squeeze(-1)
        if len(sent_logits) > 1:
            sent_logits.squeeze(-1)
        loss_fn1 = torch.nn.BCEWithLogitsLoss(reduce=False, size_average=False)
        # 去除sent_mask
        # sent_logits = sent_logits * sent_mask.float()
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sent_lbs = sent_lbs[:, 0:context_maxlen]
            # sent_weight = sent_weight[:, 0:context_maxlen]
            sent_loss = loss_fn1(sent_logits, sent_lbs.float())
            # sent_loss = (sent_loss * sent_mask.float()) * sent_weight
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


class BertForQuestionAnsweringQANetTwoCrossAttention(BertModel):
    def __init__(self, config):
        super(BertForQuestionAnsweringQANetTwoCrossAttention, self).__init__(config)
        self.bert = BertModel(config)
        self.cross_attention1 = CrossAttention(config=config)
        self.cross_attention2 = CrossAttention(config=config)
        # self.true_coattention = MyQANet(d_model=config.hidden_size)
        self.start_logits = nn.Linear(config.hidden_size, 1)
        self.end_logits = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sent = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                cls_weight=None,
                entity_ids=None,
                pq_end_pos=None,
                start_positions=None,
                end_positions=None,
                sent_mask=None,
                sent_lbs=None,
                sent_weight=None):
        if len(input_ids.shape) < 2:
            input_ids = input_ids.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            pq_end_pos = pq_end_pos.unsqueeze(0)
            if start_positions is not None and len(start_positions.shape) < 2:
                start_positions = start_positions.unsqueeze(0)
                end_positions = end_positions.unsqueeze(0)
                sent_mask = sent_mask.unsqueeze(0)
                sent_lbs = sent_lbs.unsqueeze(0)
                sent_weight = sent_weight.unsqueeze(0)
        # roberta 取消了NSP任务，所以token_type_ids没有作用
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        # add cross attention
        context_sequence_output, ques_sequence_output, context_attention_mask, ques_attention_mask = split_context_ques(
            sequence_output, pq_end_pos)
        cross_output1 = self.cross_attention1(ques_sequence_output, sequence_output, ques_attention_mask)
        cross_output2 = self.cross_attention2(ques_sequence_output, sequence_output, ques_attention_mask)
        sequence_output = cross_output1 * 0.5 + cross_output2 * 0.5
        sequence_output = sequence_output + outputs[0]
        ones_mask = torch.ones_like(attention_mask).cuda()
        context_mask = (ones_mask - token_type_ids) * attention_mask
        extended_context_mask = (1.0 - context_mask) * -10000.0
        start_logits = self.start_logits(sequence_output).squeeze(-1) + extended_context_mask
        end_logits = self.end_logits(sequence_output).squeeze(-1) + extended_context_mask
        # 去除context mask
        sent_logits = self.sent(sequence_output).squeeze(-1) * context_mask.float()
        # sent_logits = self.sent(sequence_output).squeeze(-1)
        if len(sent_logits) > 1:
            sent_logits.squeeze(-1)
        loss_fn1 = torch.nn.BCEWithLogitsLoss(reduce=False, size_average=False)
        # 去除sent_mask
        # sent_logits = sent_logits * sent_mask.float()
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sent_lbs = sent_lbs[:, 0:context_maxlen]
            # sent_weight = sent_weight[:, 0:context_maxlen]
            sent_loss = loss_fn1(sent_logits, sent_lbs.float())
            # sent_loss = (sent_loss * sent_mask.float()) * sent_weight
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


class BertForQuestionAnsweringQANetAttentionWeight(BertModel):
    def __init__(self, config):
        super(BertForQuestionAnsweringQANetAttentionWeight, self).__init__(config)
        self.bert = BertModel(config)
        self.coattention = CoattentionModel(config=config)
        self.cross_attention = CrossAttention(config=config)
        self.coattention_avg_pool = nn.AvgPool1d(512)
        self.cross_avg_pool = nn.AvgPool1d(512)
        self.coattention_linear = nn.Linear(config.hidden_size, 1)
        self.cross_attention_linear = nn.Linear(config.hidden_size, 1)
        # self.co_act = gelu
        # self.cross_act = gelu
        # self.w1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # self.w2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # # initialization
        # self.w1.data.fill_(0.5)
        # self.w2.data.fill_(0.5)
        # self.w3.data.fill_(0.3)
        self.start_logits = nn.Linear(config.hidden_size, 1)
        self.end_logits = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sent = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                cls_weight=None,
                entity_ids=None,
                pq_end_pos=None,
                start_positions=None,
                end_positions=None,
                sent_mask=None,
                sent_lbs=None,
                sent_weight=None):
        if len(input_ids.shape) < 2:
            input_ids = input_ids.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            pq_end_pos = pq_end_pos.unsqueeze(0)
            if start_positions is not None and len(start_positions.shape) < 2:
                start_positions = start_positions.unsqueeze(0)
                end_positions = end_positions.unsqueeze(0)
                sent_mask = sent_mask.unsqueeze(0)
                sent_lbs = sent_lbs.unsqueeze(0)
                sent_weight = sent_weight.unsqueeze(0)
        # roberta 取消了NSP任务，所以token_type_ids没有作用
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        # add cross attention
        context_sequence_output, ques_sequence_output, context_attention_mask, ques_attention_mask = split_context_ques(
            sequence_output, pq_end_pos)
        cross_output = self.cross_attention(ques_sequence_output, sequence_output, ques_attention_mask)
        cross_cls_output = self.cross_avg_pool(cross_output.permute(0, 2, 1)).permute(0, 2, 1).squeeze(1)
        cross_cls_attention = self.cross_attention_linear(cross_cls_output).unsqueeze(1)
        # cross_cls_attention = self.cross_act(cross_cls_attention)


        # sequence_output = self.qa_net(C=context_sequence_output,
        #                               Q=ques_sequence_output,
        #                               cmask=context_attention_mask,
        #                               qmask=ques_attention_mask)
        ones_mask = torch.ones_like(attention_mask).cuda()
        context_mask = (ones_mask - token_type_ids) * attention_mask
        ques_mask = token_type_ids * attention_mask
        coattention_mask = torch.matmul(context_mask.unsqueeze(-1).float(), ques_mask.unsqueeze(1).float())
        sequence_output = self.coattention(sequence_output, sequence_output, coattention_mask)
        sequence_cls_output = self.coattention_avg_pool(sequence_output.permute(0, 2, 1)).permute(0, 2, 1).squeeze(1)
        coattention_cls_attention = self.coattention_linear(sequence_cls_output).unsqueeze(1)
        # coattention_cls_attention = self.co_act(coattention_cls_attention)

        # biattention
        # bi_attention_output, memory = self.bi_attention(context_sequence_output, ques_sequence_output,
        #                                                 ques_attention_mask)
        # bi_attention_output = self.bi_attn_linear(bi_attention_output)
        # bs, context_len, hidden_dim = bi_attention_output.shape
        # zeros = torch.zeros((bs, 512 - context_len, hidden_dim)).cuda()
        # bi_attention_output = torch.cat([bi_attention_output, zeros], dim=1)

        # bs, context_len, hidden_dim = sequence_output.shape
        # zeros = torch.zeros((bs, 512 - context_len, hidden_dim)).cuda()
        # sequence_output = torch.cat([sequence_output, zeros], dim=1)
        # sequence_output = self.w1 * sequence_output + self.w2 * cross_output + outputs[0] + self.w3 * bi_attention_output
        # sequence_output = self.w1 * sequence_output + self.w2 * cross_output + outputs[0]
        # sequence_output = 0.5 * sequence_output + 0.5 * cross_output + outputs[0]
        # sequence_output = sequence_output + outputs[0]
        sequence_output = cross_output * cross_cls_attention + sequence_output * coattention_cls_attention
        sequence_output = sequence_output + outputs[0]
        ones_mask = torch.ones_like(attention_mask).cuda()
        context_mask = (ones_mask - token_type_ids) * attention_mask
        extended_context_mask = (1.0 - context_mask) * -10000.0
        start_logits = self.start_logits(sequence_output).squeeze(-1) + extended_context_mask
        end_logits = self.end_logits(sequence_output).squeeze(-1) + extended_context_mask
        # 去除context mask
        sent_logits = self.sent(sequence_output).squeeze(-1) * context_mask.float()
        # sent_logits = self.sent(sequence_output).squeeze(-1)
        if len(sent_logits) > 1:
            sent_logits.squeeze(-1)
        loss_fn1 = torch.nn.BCEWithLogitsLoss(reduce=False, size_average=False)
        # 去除sent_mask
        # sent_logits = sent_logits * sent_mask.float()
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sent_lbs = sent_lbs[:, 0:context_maxlen]
            # sent_weight = sent_weight[:, 0:context_maxlen]
            sent_loss = loss_fn1(sent_logits, sent_lbs.float())
            # sent_loss = (sent_loss * sent_mask.float()) * sent_weight
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


class ElectraForQuestionAnsweringQANet(ElectraModel):
    def __init__(self, config):
        super(ElectraForQuestionAnsweringQANet, self).__init__(config)
        self.electra = ElectraModel(config)
        self.coattention = CoattentionModel(config=config)
        self.cross_attention = CrossAttention(config=config)
        self.w1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # initialization
        self.w1.data.fill_(0.5)
        self.w2.data.fill_(0.5)
        # self.w3.data.fill_(0.3)
        self.start_logits = nn.Linear(config.hidden_size, 1)
        self.end_logits = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sent = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                cls_weight=None,
                entity_ids=None,
                pq_end_pos=None,
                start_positions=None,
                end_positions=None,
                sent_mask=None,
                sent_lbs=None,
                sent_weight=None):
        if len(input_ids.shape) < 2:
            input_ids = input_ids.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            pq_end_pos = pq_end_pos.unsqueeze(0)
            if start_positions is not None and len(start_positions.shape) < 2:
                start_positions = start_positions.unsqueeze(0)
                end_positions = end_positions.unsqueeze(0)
                sent_mask = sent_mask.unsqueeze(0)
                sent_lbs = sent_lbs.unsqueeze(0)
                sent_weight = sent_weight.unsqueeze(0)
        # roberta 取消了NSP任务，所以token_type_ids没有作用
        outputs = self.electra(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        # add cross attention
        context_sequence_output, ques_sequence_output, context_attention_mask, ques_attention_mask = split_context_ques(
            sequence_output, pq_end_pos)
        cross_output = self.cross_attention(ques_sequence_output, sequence_output, ques_attention_mask)

        ones_mask = torch.ones_like(attention_mask).cuda()
        context_mask = (ones_mask - token_type_ids) * attention_mask
        ques_mask = token_type_ids * attention_mask
        coattention_mask = torch.matmul(context_mask.unsqueeze(-1).float(), ques_mask.unsqueeze(1).float())
        sequence_output = self.coattention(sequence_output, sequence_output, coattention_mask)
        sequence_output = self.w1 * sequence_output + self.w2 * cross_output + outputs[0]
        ones_mask = torch.ones_like(attention_mask).cuda()
        context_mask = (ones_mask - token_type_ids) * attention_mask
        extended_context_mask = (1.0 - context_mask) * -10000.0
        start_logits = self.start_logits(sequence_output).squeeze(-1) + extended_context_mask
        end_logits = self.end_logits(sequence_output).squeeze(-1) + extended_context_mask
        # 去除context mask
        sent_logits = self.sent(sequence_output).squeeze(-1) * context_mask.float()
        if len(sent_logits) > 1:
            sent_logits.squeeze(-1)
        loss_fn1 = torch.nn.BCEWithLogitsLoss(reduce=False, size_average=False)
        # 去除sent_mask
        # sent_logits = sent_logits * sent_mask.float()
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sent_lbs = sent_lbs[:, 0:context_maxlen]
            # sent_weight = sent_weight[:, 0:context_maxlen]
            sent_loss = loss_fn1(sent_logits, sent_lbs.float())
            # sent_loss = (sent_loss * sent_mask.float()) * sent_weight
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


class AlbertForQuestionAnsweringQANet(AlbertModel):
    def __init__(self, config):
        super(AlbertForQuestionAnsweringQANet, self).__init__(config)
        self.albert = AlbertModel(config)
        self.coattention = CoattentionModel(config=config)
        self.cross_attention = CrossAttention(config=config)
        self.w1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # initialization
        self.w1.data.fill_(0.5)
        self.w2.data.fill_(0.5)
        # self.w3.data.fill_(0.3)
        self.start_logits = nn.Linear(config.hidden_size, 1)
        self.end_logits = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sent = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                cls_weight=None,
                entity_ids=None,
                pq_end_pos=None,
                start_positions=None,
                end_positions=None,
                sent_mask=None,
                sent_lbs=None,
                sent_weight=None):
        if len(input_ids.shape) < 2:
            input_ids = input_ids.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            pq_end_pos = pq_end_pos.unsqueeze(0)
            if start_positions is not None and len(start_positions.shape) < 2:
                start_positions = start_positions.unsqueeze(0)
                end_positions = end_positions.unsqueeze(0)
                sent_mask = sent_mask.unsqueeze(0)
                sent_lbs = sent_lbs.unsqueeze(0)
                sent_weight = sent_weight.unsqueeze(0)
        # roberta 取消了NSP任务，所以token_type_ids没有作用
        outputs = self.albert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        # add cross attention
        context_sequence_output, ques_sequence_output, context_attention_mask, ques_attention_mask = split_context_ques(
            sequence_output, pq_end_pos)
        cross_output = self.cross_attention(ques_sequence_output, sequence_output, ques_attention_mask)

        ones_mask = torch.ones_like(attention_mask).cuda()
        context_mask = (ones_mask - token_type_ids) * attention_mask
        ques_mask = token_type_ids * attention_mask
        coattention_mask = torch.matmul(context_mask.unsqueeze(-1).float(), ques_mask.unsqueeze(1).float())
        sequence_output = self.coattention(sequence_output, sequence_output, coattention_mask)
        sequence_output = self.w1 * sequence_output + self.w2 * cross_output + outputs[0]
        ones_mask = torch.ones_like(attention_mask).cuda()
        context_mask = (ones_mask - token_type_ids) * attention_mask
        extended_context_mask = (1.0 - context_mask) * -10000.0
        start_logits = self.start_logits(sequence_output).squeeze(-1) + extended_context_mask
        end_logits = self.end_logits(sequence_output).squeeze(-1) + extended_context_mask
        # 去除context mask
        sent_logits = self.sent(sequence_output).squeeze(-1) * context_mask.float()
        if len(sent_logits) > 1:
            sent_logits.squeeze(-1)
        loss_fn1 = torch.nn.BCEWithLogitsLoss(reduce=False, size_average=False)
        # 去除sent_mask
        # sent_logits = sent_logits * sent_mask.float()
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sent_lbs = sent_lbs[:, 0:context_maxlen]
            # sent_weight = sent_weight[:, 0:context_maxlen]
            sent_loss = loss_fn1(sent_logits, sent_lbs.float())
            # sent_loss = (sent_loss * sent_mask.float()) * sent_weight
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


class AlbertForQuestionAnsweringQANetWithMask(AlbertModel):
    def __init__(self, config):
        super(AlbertForQuestionAnsweringQANetWithMask, self).__init__(config)
        self.albert = AlbertModel(config)
        self.coattention = CoattentionModel(config=config)
        self.cross_attention = CrossAttention(config=config)
        self.w1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # initialization
        self.w1.data.fill_(0.5)
        self.w2.data.fill_(0.5)
        # self.w3.data.fill_(0.3)
        self.start_logits = nn.Linear(config.hidden_size, 1)
        self.end_logits = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sent = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                cls_weight=None,
                entity_ids=None,
                pq_end_pos=None,
                start_positions=None,
                end_positions=None,
                sent_mask=None,
                sent_lbs=None,
                sent_weight=None):
        if len(input_ids.shape) < 2:
            input_ids = input_ids.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            pq_end_pos = pq_end_pos.unsqueeze(0)
            if start_positions is not None and len(start_positions.shape) < 2:
                start_positions = start_positions.unsqueeze(0)
                end_positions = end_positions.unsqueeze(0)
                sent_mask = sent_mask.unsqueeze(0)
                sent_lbs = sent_lbs.unsqueeze(0)
                sent_weight = sent_weight.unsqueeze(0)
        # roberta 取消了NSP任务，所以token_type_ids没有作用
        outputs = self.albert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        import pdb; pdb.set_trace()
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        # add cross attention
        context_sequence_output, ques_sequence_output, context_attention_mask, ques_attention_mask = split_context_ques(
            sequence_output, pq_end_pos)
        cross_output = self.cross_attention(ques_sequence_output, sequence_output, ques_attention_mask)

        ones_mask = torch.ones_like(attention_mask).cuda()
        context_mask = (ones_mask - token_type_ids) * attention_mask
        ques_mask = token_type_ids * attention_mask
        coattention_mask = torch.matmul(context_mask.unsqueeze(-1).float(), ques_mask.unsqueeze(1).float())
        sequence_output = self.coattention(sequence_output, sequence_output, coattention_mask)
        sequence_output = self.w1 * sequence_output + self.w2 * cross_output + outputs[0]
        ones_mask = torch.ones_like(attention_mask).cuda()
        context_mask = (ones_mask - token_type_ids) * attention_mask
        extended_context_mask = (1.0 - context_mask) * -10000.0
        start_logits = self.start_logits(sequence_output).squeeze(-1) + extended_context_mask
        end_logits = self.end_logits(sequence_output).squeeze(-1) + extended_context_mask
        # 去除context mask
        sent_logits = self.sent(sequence_output).squeeze(-1) * context_mask.float()
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
            # sent_loss = (sent_loss * sent_mask.float()) * sent_weight
            sent_loss = (sent_loss * sent_mask.float())
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


class ElectraForQuestionAnsweringQANetWoCro(ElectraModel):
    def __init__(self, config):
        super(ElectraForQuestionAnsweringQANetWoCro, self).__init__(config)
        self.electra = ElectraModel(config)
        # self.coattention = CoattentionModel(config=config)
        self.cross_attention = CrossAttention(config=config)
        # self.w1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # initialization
        # self.w1.data.fill_(0.5)
        self.w2.data.fill_(1)
        # self.w3.data.fill_(0.3)
        self.start_logits = nn.Linear(config.hidden_size, 1)
        self.end_logits = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sent = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                cls_weight=None,
                entity_ids=None,
                pq_end_pos=None,
                start_positions=None,
                end_positions=None,
                sent_mask=None,
                sent_lbs=None,
                sent_weight=None):
        if len(input_ids.shape) < 2:
            input_ids = input_ids.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            pq_end_pos = pq_end_pos.unsqueeze(0)
            if start_positions is not None and len(start_positions.shape) < 2:
                start_positions = start_positions.unsqueeze(0)
                end_positions = end_positions.unsqueeze(0)
                sent_mask = sent_mask.unsqueeze(0)
                sent_lbs = sent_lbs.unsqueeze(0)
                sent_weight = sent_weight.unsqueeze(0)
        # roberta 取消了NSP任务，所以token_type_ids没有作用
        outputs = self.electra(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        # add cross attention
        context_sequence_output, ques_sequence_output, context_attention_mask, ques_attention_mask = split_context_ques(
            sequence_output, pq_end_pos)
        cross_output = self.cross_attention(ques_sequence_output, sequence_output, ques_attention_mask)

        ones_mask = torch.ones_like(attention_mask).cuda()
        context_mask = (ones_mask - token_type_ids) * attention_mask
        ques_mask = token_type_ids * attention_mask
        coattention_mask = torch.matmul(context_mask.unsqueeze(-1).float(), ques_mask.unsqueeze(1).float())
        # sequence_output = self.coattention(sequence_output, sequence_output, coattention_mask)
        sequence_output = self.w2 * cross_output + outputs[0]
        ones_mask = torch.ones_like(attention_mask).cuda()
        context_mask = (ones_mask - token_type_ids) * attention_mask
        extended_context_mask = (1.0 - context_mask) * -10000.0
        start_logits = self.start_logits(sequence_output).squeeze(-1) + extended_context_mask
        end_logits = self.end_logits(sequence_output).squeeze(-1) + extended_context_mask
        # 去除context mask
        sent_logits = self.sent(sequence_output).squeeze(-1) * context_mask.float()
        if len(sent_logits) > 1:
            sent_logits.squeeze(-1)
        loss_fn1 = torch.nn.BCEWithLogitsLoss(reduce=False, size_average=False)
        # 去除sent_mask
        # sent_logits = sent_logits * sent_mask.float()
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sent_lbs = sent_lbs[:, 0:context_maxlen]
            # sent_weight = sent_weight[:, 0:context_maxlen]
            sent_loss = loss_fn1(sent_logits, sent_lbs.float())
            # sent_loss = (sent_loss * sent_mask.float()) * sent_weight
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


class ElectraForQuestionAnsweringQANetWoLN(ElectraModel):
    def __init__(self, config):
        super(ElectraForQuestionAnsweringQANetWoLN, self).__init__(config)
        self.electra = ElectraModel(config)
        self.coattention = CoattentionModel(config=config)
        # self.cross_attention = CrossAttention(config=config)
        self.w1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # self.w2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # initialization
        self.w1.data.fill_(1.0)
        # self.w2.data.fill_(0.5)
        # self.w3.data.fill_(0.3)
        self.start_logits = nn.Linear(config.hidden_size, 1)
        self.end_logits = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sent = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                cls_weight=None,
                entity_ids=None,
                pq_end_pos=None,
                start_positions=None,
                end_positions=None,
                sent_mask=None,
                sent_lbs=None,
                sent_weight=None):
        if len(input_ids.shape) < 2:
            input_ids = input_ids.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            pq_end_pos = pq_end_pos.unsqueeze(0)
            if start_positions is not None and len(start_positions.shape) < 2:
                start_positions = start_positions.unsqueeze(0)
                end_positions = end_positions.unsqueeze(0)
                sent_mask = sent_mask.unsqueeze(0)
                sent_lbs = sent_lbs.unsqueeze(0)
                sent_weight = sent_weight.unsqueeze(0)
        # roberta 取消了NSP任务，所以token_type_ids没有作用
        outputs = self.electra(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        # add cross attention
        context_sequence_output, ques_sequence_output, context_attention_mask, ques_attention_mask = split_context_ques(
            sequence_output, pq_end_pos)
        # cross_output = self.cross_attention(ques_sequence_output, sequence_output, ques_attention_mask)

        ones_mask = torch.ones_like(attention_mask).cuda()
        context_mask = (ones_mask - token_type_ids) * attention_mask
        ques_mask = token_type_ids * attention_mask
        coattention_mask = torch.matmul(context_mask.unsqueeze(-1).float(), ques_mask.unsqueeze(1).float())
        sequence_output = self.coattention(sequence_output, sequence_output, coattention_mask)
        sequence_output = self.w1 * sequence_output + outputs[0]
        ones_mask = torch.ones_like(attention_mask).cuda()
        context_mask = (ones_mask - token_type_ids) * attention_mask
        extended_context_mask = (1.0 - context_mask) * -10000.0
        start_logits = self.start_logits(sequence_output).squeeze(-1) + extended_context_mask
        end_logits = self.end_logits(sequence_output).squeeze(-1) + extended_context_mask
        # 去除context mask
        sent_logits = self.sent(sequence_output).squeeze(-1) * context_mask.float()
        if len(sent_logits) > 1:
            sent_logits.squeeze(-1)
        loss_fn1 = torch.nn.BCEWithLogitsLoss(reduce=False, size_average=False)
        # 去除sent_mask
        # sent_logits = sent_logits * sent_mask.float()
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sent_lbs = sent_lbs[:, 0:context_maxlen]
            # sent_weight = sent_weight[:, 0:context_maxlen]
            sent_loss = loss_fn1(sent_logits, sent_lbs.float())
            # sent_loss = (sent_loss * sent_mask.float()) * sent_weight
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


class ElectraForQuestionAnsweringDivideNet(ElectraModel):
    def __init__(self, config):
        super(ElectraForQuestionAnsweringDivideNet, self).__init__(config)
        self.electra = ElectraModel(config)
        self.coattention = CoattentionModel(config=config)
        self.cross_attention = CrossAttention(config=config)
        self.coattention2 = CoattentionModel(config=config)
        self.cross_attention2 = CrossAttention(config=config)
        self.w1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w4 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # initialization
        self.w1.data.fill_(0.5)
        self.w2.data.fill_(0.5)
        self.w3.data.fill_(0.5)
        self.w4.data.fill_(0.5)
        # self.w3.data.fill_(0.3)
        self.start_logits = nn.Linear(config.hidden_size, 1)
        self.end_logits = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sent = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                cls_weight=None,
                entity_ids=None,
                pq_end_pos=None,
                start_positions=None,
                end_positions=None,
                sent_mask=None,
                sent_lbs=None,
                sent_weight=None):
        if len(input_ids.shape) < 2:
            input_ids = input_ids.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            pq_end_pos = pq_end_pos.unsqueeze(0)
            if start_positions is not None and len(start_positions.shape) < 2:
                start_positions = start_positions.unsqueeze(0)
                end_positions = end_positions.unsqueeze(0)
                sent_mask = sent_mask.unsqueeze(0)
                sent_lbs = sent_lbs.unsqueeze(0)
                sent_weight = sent_weight.unsqueeze(0)
        # roberta 取消了NSP任务，所以token_type_ids没有作用
        outputs = self.electra(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        # add cross attention
        context_sequence_output, ques_sequence_output, context_attention_mask, ques_attention_mask = split_context_ques(
            sequence_output, pq_end_pos)
        cross_output = self.cross_attention(ques_sequence_output, sequence_output, ques_attention_mask)
        cross_output2 = self.cross_attention2(ques_sequence_output, sequence_output, ques_attention_mask)

        ones_mask = torch.ones_like(attention_mask).cuda()
        context_mask = (ones_mask - token_type_ids) * attention_mask
        ques_mask = token_type_ids * attention_mask
        coattention_mask = torch.matmul(context_mask.unsqueeze(-1).float(), ques_mask.unsqueeze(1).float())
        coattention_output = self.coattention(sequence_output, sequence_output, coattention_mask)
        coattention_output2 = self.coattention(sequence_output, sequence_output, coattention_mask)
        sent_output = self.w1 * coattention_output + self.w2 * cross_output + outputs[0]
        ans_output = (self.w1 * coattention_output + self.w2 * cross_output
                      + self.w3 * coattention_output2 + self.w4 * cross_output2) * 0.5 + outputs[0]
        ones_mask = torch.ones_like(attention_mask).cuda()
        context_mask = (ones_mask - token_type_ids) * attention_mask
        extended_context_mask = (1.0 - context_mask) * -10000.0
        start_logits = self.start_logits(ans_output).squeeze(-1) + extended_context_mask
        end_logits = self.end_logits(ans_output).squeeze(-1) + extended_context_mask
        # 去除context mask
        sent_logits = self.sent(sent_output).squeeze(-1) * context_mask.float()
        if len(sent_logits) > 1:
            sent_logits.squeeze(-1)
        loss_fn1 = torch.nn.BCEWithLogitsLoss(reduce=False, size_average=False)
        # 去除sent_mask
        # sent_logits = sent_logits * sent_mask.float()
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sent_lbs = sent_lbs[:, 0:context_maxlen]
            # sent_weight = sent_weight[:, 0:context_maxlen]
            sent_loss = loss_fn1(sent_logits, sent_lbs.float())
            # sent_loss = (sent_loss * sent_mask.float()) * sent_weight
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


class DebertaForQuestionAnsweringQANet(DebertaModel):
    def __init__(self, config):
        super(DebertaForQuestionAnsweringQANet, self).__init__(config)
        self.deberta = DebertaModel(config)
        self.coattention = CoattentionModel(config=config)
        self.cross_attention = CrossAttention(config=config)
        self.w1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # initialization
        self.w1.data.fill_(0.5)
        self.w2.data.fill_(0.5)
        # self.w3.data.fill_(0.3)
        self.start_logits = nn.Linear(config.hidden_size, 1)
        self.end_logits = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sent = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                cls_weight=None,
                entity_ids=None,
                pq_end_pos=None,
                start_positions=None,
                end_positions=None,
                sent_mask=None,
                sent_lbs=None,
                sent_weight=None):
        if len(input_ids.shape) < 2:
            input_ids = input_ids.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            pq_end_pos = pq_end_pos.unsqueeze(0)
            if start_positions is not None and len(start_positions.shape) < 2:
                start_positions = start_positions.unsqueeze(0)
                end_positions = end_positions.unsqueeze(0)
                sent_mask = sent_mask.unsqueeze(0)
                sent_lbs = sent_lbs.unsqueeze(0)
                sent_weight = sent_weight.unsqueeze(0)
        # roberta 取消了NSP任务，所以token_type_ids没有作用
        outputs = self.deberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        # add cross attention
        context_sequence_output, ques_sequence_output, context_attention_mask, ques_attention_mask = split_context_ques(
            sequence_output, pq_end_pos)
        cross_output = self.cross_attention(ques_sequence_output, sequence_output, ques_attention_mask)

        ones_mask = torch.ones_like(attention_mask).cuda()
        context_mask = (ones_mask - token_type_ids) * attention_mask
        ques_mask = token_type_ids * attention_mask
        coattention_mask = torch.matmul(context_mask.unsqueeze(-1).float(), ques_mask.unsqueeze(1).float())
        sequence_output = self.coattention(sequence_output, sequence_output, coattention_mask)
        sequence_output = self.w1 * sequence_output + self.w2 * cross_output + outputs[0]
        ones_mask = torch.ones_like(attention_mask).cuda()
        context_mask = (ones_mask - token_type_ids) * attention_mask
        extended_context_mask = (1.0 - context_mask) * -10000.0
        start_logits = self.start_logits(sequence_output).squeeze(-1) + extended_context_mask
        end_logits = self.end_logits(sequence_output).squeeze(-1) + extended_context_mask
        # 去除context mask
        sent_logits = self.sent(sequence_output).squeeze(-1) * context_mask.float()
        if len(sent_logits) > 1:
            sent_logits.squeeze(-1)
        loss_fn1 = torch.nn.BCEWithLogitsLoss(reduce=False, size_average=False)
        # 去除sent_mask
        # sent_logits = sent_logits * sent_mask.float()
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sent_lbs = sent_lbs[:, 0:context_maxlen]
            # sent_weight = sent_weight[:, 0:context_maxlen]
            sent_loss = loss_fn1(sent_logits, sent_lbs.float())
            # sent_loss = (sent_loss * sent_mask.float()) * sent_weight
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


class ElectraForQuestionAnsweringQANetWithSentWeight(ElectraModel):
    def __init__(self, config):
        super(ElectraForQuestionAnsweringQANetWithSentWeight, self).__init__(config)
        self.electra = ElectraModel(config)
        self.coattention = CoattentionModel(config=config)
        self.cross_attention = CrossAttention(config=config)
        self.w1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # initialization
        self.w1.data.fill_(0.5)
        self.w2.data.fill_(0.5)
        # self.w3.data.fill_(0.3)
        self.start_logits = nn.Linear(config.hidden_size, 1)
        self.end_logits = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sent = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                cls_weight=None,
                entity_ids=None,
                pq_end_pos=None,
                start_positions=None,
                end_positions=None,
                sent_mask=None,
                sent_lbs=None,
                sent_weight=None):
        if len(input_ids.shape) < 2:
            input_ids = input_ids.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            pq_end_pos = pq_end_pos.unsqueeze(0)
            if start_positions is not None and len(start_positions.shape) < 2:
                start_positions = start_positions.unsqueeze(0)
                end_positions = end_positions.unsqueeze(0)
                sent_mask = sent_mask.unsqueeze(0)
                sent_lbs = sent_lbs.unsqueeze(0)
                sent_weight = sent_weight.unsqueeze(0)
        # roberta 取消了NSP任务，所以token_type_ids没有作用
        outputs = self.electra(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        # add cross attention
        context_sequence_output, ques_sequence_output, context_attention_mask, ques_attention_mask = split_context_ques(
            sequence_output, pq_end_pos)
        cross_output = self.cross_attention(ques_sequence_output, sequence_output, ques_attention_mask)

        ones_mask = torch.ones_like(attention_mask).cuda()
        context_mask = (ones_mask - token_type_ids) * attention_mask
        ques_mask = token_type_ids * attention_mask
        coattention_mask = torch.matmul(context_mask.unsqueeze(-1).float(), ques_mask.unsqueeze(1).float())
        sequence_output = self.coattention(sequence_output, sequence_output, coattention_mask)
        sequence_output = self.w1 * sequence_output + self.w2 * cross_output + outputs[0]
        ones_mask = torch.ones_like(attention_mask).cuda()
        context_mask = (ones_mask - token_type_ids) * attention_mask
        extended_context_mask = (1.0 - context_mask) * -10000.0
        start_logits = self.start_logits(sequence_output).squeeze(-1) + extended_context_mask
        end_logits = self.end_logits(sequence_output).squeeze(-1) + extended_context_mask
        # 去除context mask
        sent_logits = self.sent(sequence_output).squeeze(-1) * context_mask.float()
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
            # sent_loss = (sent_loss * sent_mask.float()) * sent_weight
            sent_loss = (sent_loss * sent_mask.float())
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


class ElectraForQuestionAnsweringQANetDouble(ElectraModel):
    def __init__(self, config):
        super(ElectraForQuestionAnsweringQANetDouble, self).__init__(config)
        self.electra = ElectraModel(config)
        self.coattention1 = CoattentionModel(config=config)
        self.cross_attention1 = CrossAttention(config=config)
        self.coattention2 = CoattentionModel(config=config)
        self.cross_attention2 = CrossAttention(config=config)
        self.w1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w4 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # initialization
        self.w1.data.fill_(0.25)
        self.w2.data.fill_(0.25)
        self.w3.data.fill_(0.25)
        self.w4.data.fill_(0.25)
        self.start_logits = nn.Linear(config.hidden_size, 1)
        self.end_logits = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sent = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                cls_weight=None,
                entity_ids=None,
                pq_end_pos=None,
                start_positions=None,
                end_positions=None,
                sent_mask=None,
                sent_lbs=None,
                sent_weight=None):
        if len(input_ids.shape) < 2:
            input_ids = input_ids.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            pq_end_pos = pq_end_pos.unsqueeze(0)
            if start_positions is not None and len(start_positions.shape) < 2:
                start_positions = start_positions.unsqueeze(0)
                end_positions = end_positions.unsqueeze(0)
                sent_mask = sent_mask.unsqueeze(0)
                sent_lbs = sent_lbs.unsqueeze(0)
                sent_weight = sent_weight.unsqueeze(0)
        # roberta 取消了NSP任务，所以token_type_ids没有作用
        outputs = self.electra(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        # add cross attention
        context_sequence_output, ques_sequence_output, context_attention_mask, ques_attention_mask = split_context_ques(
            sequence_output, pq_end_pos)
        cross_output1 = self.cross_attention1(ques_sequence_output, sequence_output, ques_attention_mask)
        cross_output2 = self.cross_attention2(ques_sequence_output, sequence_output, ques_attention_mask)

        ones_mask = torch.ones_like(attention_mask).cuda()
        context_mask = (ones_mask - token_type_ids) * attention_mask
        ques_mask = token_type_ids * attention_mask
        coattention_mask = torch.matmul(context_mask.unsqueeze(-1).float(), ques_mask.unsqueeze(1).float())
        co_output1 = self.coattention1(sequence_output, sequence_output, coattention_mask)
        co_output2 = self.coattention2(sequence_output, sequence_output, coattention_mask)
        sequence_output = self.w1 * cross_output1 + self.w2 * cross_output2 + self.w3 * co_output1 + self.w4 * co_output2
        sequence_output = sequence_output + outputs[0]
        ones_mask = torch.ones_like(attention_mask).cuda()
        context_mask = (ones_mask - token_type_ids) * attention_mask
        extended_context_mask = (1.0 - context_mask) * -10000.0
        start_logits = self.start_logits(sequence_output).squeeze(-1) + extended_context_mask
        end_logits = self.end_logits(sequence_output).squeeze(-1) + extended_context_mask
        # 去除context mask
        sent_logits = self.sent(sequence_output).squeeze(-1) * context_mask.float()
        # sent_logits = self.sent(sequence_output).squeeze(-1)
        if len(sent_logits) > 1:
            sent_logits.squeeze(-1)
        loss_fn1 = torch.nn.BCEWithLogitsLoss(reduce=False, size_average=False)
        # 去除sent_mask
        # sent_logits = sent_logits * sent_mask.float()
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sent_lbs = sent_lbs[:, 0:context_maxlen]
            # sent_weight = sent_weight[:, 0:context_maxlen]
            sent_loss = loss_fn1(sent_logits, sent_lbs.float())
            # sent_loss = (sent_loss * sent_mask.float()) * sent_weight
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


class ElectraForQuestionAnsweringQANetDoubleCan(ElectraModel):
    def __init__(self, config):
        super(ElectraForQuestionAnsweringQANetDoubleCan, self).__init__(config)
        self.electra = ElectraModel(config)
        self.coattention1 = CoattentionModel(config=config)
        self.cross_attention1 = CrossAttention(config=config)
        self.coattention2 = CoattentionModel(config=config)
        self.cross_attention2 = CrossAttention(config=config)
        self.can_linear = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.w1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w4 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # initialization
        self.w1.data.fill_(0.5)
        self.w2.data.fill_(0.5)
        self.w3.data.fill_(0.5)
        self.w4.data.fill_(0.5)
        self.start_logits = nn.Linear(config.hidden_size, 1)
        self.end_logits = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sent = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                cls_weight=None,
                entity_ids=None,
                pq_end_pos=None,
                start_positions=None,
                end_positions=None,
                sent_mask=None,
                sent_lbs=None,
                sent_weight=None):
        if len(input_ids.shape) < 2:
            input_ids = input_ids.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            pq_end_pos = pq_end_pos.unsqueeze(0)
            if start_positions is not None and len(start_positions.shape) < 2:
                start_positions = start_positions.unsqueeze(0)
                end_positions = end_positions.unsqueeze(0)
                sent_mask = sent_mask.unsqueeze(0)
                sent_lbs = sent_lbs.unsqueeze(0)
                sent_weight = sent_weight.unsqueeze(0)
        # roberta 取消了NSP任务，所以token_type_ids没有作用
        outputs = self.electra(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        # add cross attention
        context_sequence_output, ques_sequence_output, context_attention_mask, ques_attention_mask = split_context_ques(
            sequence_output, pq_end_pos)
        cross_output1 = self.cross_attention1(ques_sequence_output, sequence_output, ques_attention_mask)
        cross_output2 = self.cross_attention2(ques_sequence_output, sequence_output, ques_attention_mask)

        ones_mask = torch.ones_like(attention_mask).cuda()
        context_mask = (ones_mask - token_type_ids) * attention_mask
        ques_mask = token_type_ids * attention_mask
        coattention_mask = torch.matmul(context_mask.unsqueeze(-1).float(), ques_mask.unsqueeze(1).float())
        co_output1 = self.coattention1(sequence_output, sequence_output, coattention_mask)
        co_output2 = self.coattention2(sequence_output, sequence_output, coattention_mask)
        sequence_output1 = self.w1 * cross_output1 + self.w2 * co_output1
        sequence_output2 = torch.cat([cross_output2, co_output2], dim=2)
        sequence_output2 = self.can_linear(sequence_output2)
        sequence_output = self.w3 * sequence_output1 + self.w4 * sequence_output2
        sequence_output = sequence_output + outputs[0]
        ones_mask = torch.ones_like(attention_mask).cuda()
        context_mask = (ones_mask - token_type_ids) * attention_mask
        extended_context_mask = (1.0 - context_mask) * -10000.0
        start_logits = self.start_logits(sequence_output).squeeze(-1) + extended_context_mask
        end_logits = self.end_logits(sequence_output).squeeze(-1) + extended_context_mask
        # 去除context mask
        sent_logits = self.sent(sequence_output).squeeze(-1) * context_mask.float()
        # sent_logits = self.sent(sequence_output).squeeze(-1)
        if len(sent_logits) > 1:
            sent_logits.squeeze(-1)
        loss_fn1 = torch.nn.BCEWithLogitsLoss(reduce=False, size_average=False)
        # 去除sent_mask
        # sent_logits = sent_logits * sent_mask.float()
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sent_lbs = sent_lbs[:, 0:context_maxlen]
            # sent_weight = sent_weight[:, 0:context_maxlen]
            sent_loss = loss_fn1(sent_logits, sent_lbs.float())
            # sent_loss = (sent_loss * sent_mask.float()) * sent_weight
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


class BertForQuestionAnsweringQANetDoubleCan(BertModel):
    def __init__(self, config):
        super(BertForQuestionAnsweringQANetDoubleCan, self).__init__(config)
        self.bert = BertModel(config)
        self.coattention1 = CoattentionModel(config=config)
        self.cross_attention1 = CrossAttention(config=config)
        self.coattention2 = CoattentionModel(config=config)
        self.cross_attention2 = CrossAttention(config=config)
        self.can_linear = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.w1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w4 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # initialization
        self.w1.data.fill_(0.5)
        self.w2.data.fill_(0.5)
        self.w3.data.fill_(0.5)
        self.w4.data.fill_(0.5)
        self.start_logits = nn.Linear(config.hidden_size, 1)
        self.end_logits = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sent = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                cls_weight=None,
                entity_ids=None,
                pq_end_pos=None,
                start_positions=None,
                end_positions=None,
                sent_mask=None,
                sent_lbs=None,
                sent_weight=None):
        if len(input_ids.shape) < 2:
            input_ids = input_ids.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            pq_end_pos = pq_end_pos.unsqueeze(0)
            if start_positions is not None and len(start_positions.shape) < 2:
                start_positions = start_positions.unsqueeze(0)
                end_positions = end_positions.unsqueeze(0)
                sent_mask = sent_mask.unsqueeze(0)
                sent_lbs = sent_lbs.unsqueeze(0)
                sent_weight = sent_weight.unsqueeze(0)
        # roberta 取消了NSP任务，所以token_type_ids没有作用
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        # add cross attention
        context_sequence_output, ques_sequence_output, context_attention_mask, ques_attention_mask = split_context_ques(
            sequence_output, pq_end_pos)
        cross_output1 = self.cross_attention1(ques_sequence_output, sequence_output, ques_attention_mask)
        cross_output2 = self.cross_attention2(ques_sequence_output, sequence_output, ques_attention_mask)

        ones_mask = torch.ones_like(attention_mask).cuda()
        context_mask = (ones_mask - token_type_ids) * attention_mask
        ques_mask = token_type_ids * attention_mask
        coattention_mask = torch.matmul(context_mask.unsqueeze(-1).float(), ques_mask.unsqueeze(1).float())
        co_output1 = self.coattention1(sequence_output, sequence_output, coattention_mask)
        co_output2 = self.coattention2(sequence_output, sequence_output, coattention_mask)
        sequence_output1 = self.w1 * cross_output1 + self.w2 * co_output1
        sequence_output2 = torch.cat([cross_output2, co_output2], dim=2)
        sequence_output2 = self.can_linear(sequence_output2)
        sequence_output = self.w3 * sequence_output1 + self.w4 * sequence_output2
        sequence_output = sequence_output + outputs[0]
        ones_mask = torch.ones_like(attention_mask).cuda()
        context_mask = (ones_mask - token_type_ids) * attention_mask
        extended_context_mask = (1.0 - context_mask) * -10000.0
        start_logits = self.start_logits(sequence_output).squeeze(-1) + extended_context_mask
        end_logits = self.end_logits(sequence_output).squeeze(-1) + extended_context_mask
        # 去除context mask
        sent_logits = self.sent(sequence_output).squeeze(-1) * context_mask.float()
        # sent_logits = self.sent(sequence_output).squeeze(-1)
        if len(sent_logits) > 1:
            sent_logits.squeeze(-1)
        loss_fn1 = torch.nn.BCEWithLogitsLoss(reduce=False, size_average=False)
        # 去除sent_mask
        # sent_logits = sent_logits * sent_mask.float()
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sent_lbs = sent_lbs[:, 0:context_maxlen]
            # sent_weight = sent_weight[:, 0:context_maxlen]
            sent_loss = loss_fn1(sent_logits, sent_lbs.float())
            # sent_loss = (sent_loss * sent_mask.float()) * sent_weight
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


class ElectraForQuestionAnsweringTwoCrossAttention(ElectraModel):
    def __init__(self, config):
        super(ElectraForQuestionAnsweringTwoCrossAttention, self).__init__(config)
        self.electra = ElectraModel(config)
        self.cross_attention1 = CrossAttention(config=config)
        self.cross_attention2 = CrossAttention(config=config)
        # self.bi_attention = BiAttention(config.hidden_size, config.hidden_size, config.hidden_size)
        # self.bi_attn_linear = nn.Linear(config.hidden_size * 4, config.hidden_size)
        # self.bi_attn_dropout = nn.Dropout(0.2)
        self.w1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # self.w3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # initialization
        self.w1.data.fill_(0.5)
        self.w2.data.fill_(0.5)
        # self.w3.data.fill_(0.3)
        self.start_logits = nn.Linear(config.hidden_size, 1)
        self.end_logits = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sent = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                cls_weight=None,
                entity_ids=None,
                pq_end_pos=None,
                start_positions=None,
                end_positions=None,
                sent_mask=None,
                sent_lbs=None,
                sent_weight=None):
        if len(input_ids.shape) < 2:
            input_ids = input_ids.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            pq_end_pos = pq_end_pos.unsqueeze(0)
            if start_positions is not None and len(start_positions.shape) < 2:
                start_positions = start_positions.unsqueeze(0)
                end_positions = end_positions.unsqueeze(0)
                sent_mask = sent_mask.unsqueeze(0)
                sent_lbs = sent_lbs.unsqueeze(0)
                sent_weight = sent_weight.unsqueeze(0)
        # roberta 取消了NSP任务，所以token_type_ids没有作用
        outputs = self.electra(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        # add cross attention
        context_sequence_output, ques_sequence_output, context_attention_mask, ques_attention_mask = split_context_ques(
            sequence_output, pq_end_pos)
        cross_output1 = self.cross_attention1(ques_sequence_output, sequence_output, ques_attention_mask)
        cross_output2 = self.cross_attention2(ques_sequence_output, sequence_output, ques_attention_mask)
        sequence_output = self.w1 * cross_output1 + self.w2 * cross_output2
        sequence_output = sequence_output + outputs[0]
        ones_mask = torch.ones_like(attention_mask).cuda()
        context_mask = (ones_mask - token_type_ids) * attention_mask
        extended_context_mask = (1.0 - context_mask) * -10000.0
        start_logits = self.start_logits(sequence_output).squeeze(-1) + extended_context_mask
        end_logits = self.end_logits(sequence_output).squeeze(-1) + extended_context_mask
        # 去除context mask
        sent_logits = self.sent(sequence_output).squeeze(-1) * context_mask.float()
        # sent_logits = self.sent(sequence_output).squeeze(-1)
        if len(sent_logits) > 1:
            sent_logits.squeeze(-1)
        loss_fn1 = torch.nn.BCEWithLogitsLoss(reduce=False, size_average=False)
        # 去除sent_mask
        # sent_logits = sent_logits * sent_mask.float()
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sent_lbs = sent_lbs[:, 0:context_maxlen]
            # sent_weight = sent_weight[:, 0:context_maxlen]
            sent_loss = loss_fn1(sent_logits, sent_lbs.float())
            # sent_loss = (sent_loss * sent_mask.float()) * sent_weight
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


class ElectraForQuestionAnsweringTwoFakeCrossAttention(ElectraModel):
    def __init__(self, config):
        super(ElectraForQuestionAnsweringTwoFakeCrossAttention, self).__init__(config)
        self.electra = ElectraModel(config)
        self.fake_cross_attention1 = CoattentionModel(config=config)
        self.fake_cross_attention2 = CoattentionModel(config=config)
        self.w1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # initialization
        self.w1.data.fill_(0.5)
        self.w2.data.fill_(0.5)
        self.start_logits = nn.Linear(config.hidden_size, 1)
        self.end_logits = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sent = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                cls_weight=None,
                entity_ids=None,
                pq_end_pos=None,
                start_positions=None,
                end_positions=None,
                sent_mask=None,
                sent_lbs=None,
                sent_weight=None):
        if len(input_ids.shape) < 2:
            input_ids = input_ids.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            pq_end_pos = pq_end_pos.unsqueeze(0)
            if start_positions is not None and len(start_positions.shape) < 2:
                start_positions = start_positions.unsqueeze(0)
                end_positions = end_positions.unsqueeze(0)
                sent_mask = sent_mask.unsqueeze(0)
                sent_lbs = sent_lbs.unsqueeze(0)
                sent_weight = sent_weight.unsqueeze(0)
        # roberta 取消了NSP任务，所以token_type_ids没有作用
        outputs = self.electra(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        # add cross attention
        context_sequence_output, ques_sequence_output, context_attention_mask, ques_attention_mask = split_context_ques(
            sequence_output, pq_end_pos)

        ones_mask = torch.ones_like(attention_mask).cuda()
        context_mask = (ones_mask - token_type_ids) * attention_mask
        ques_mask = token_type_ids * attention_mask
        coattention_mask = torch.matmul(context_mask.unsqueeze(-1).float(), ques_mask.unsqueeze(1).float())
        cross_output1 = self.fake_cross_attention1(sequence_output, sequence_output, coattention_mask)
        cross_output2 = self.fake_cross_attention2(sequence_output, sequence_output, coattention_mask)
        sequence_output = self.w1 * cross_output1 + self.w2 * cross_output2
        sequence_output = sequence_output + outputs[0]
        ones_mask = torch.ones_like(attention_mask).cuda()
        context_mask = (ones_mask - token_type_ids) * attention_mask
        extended_context_mask = (1.0 - context_mask) * -10000.0
        start_logits = self.start_logits(sequence_output).squeeze(-1) + extended_context_mask
        end_logits = self.end_logits(sequence_output).squeeze(-1) + extended_context_mask
        # 去除context mask
        sent_logits = self.sent(sequence_output).squeeze(-1) * context_mask.float()
        # sent_logits = self.sent(sequence_output).squeeze(-1)
        if len(sent_logits) > 1:
            sent_logits.squeeze(-1)
        loss_fn1 = torch.nn.BCEWithLogitsLoss(reduce=False, size_average=False)
        # 去除sent_mask
        # sent_logits = sent_logits * sent_mask.float()
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sent_lbs = sent_lbs[:, 0:context_maxlen]
            # sent_weight = sent_weight[:, 0:context_maxlen]
            sent_loss = loss_fn1(sent_logits, sent_lbs.float())
            # sent_loss = (sent_loss * sent_mask.float()) * sent_weight
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


class ElectraForQuestionAnsweringQANetAttentionWeight(ElectraModel):
    def __init__(self, config):
        super(ElectraForQuestionAnsweringQANetAttentionWeight, self).__init__(config)
        self.electra = ElectraModel(config)
        self.coattention = CoattentionModel(config=config)
        self.cross_attention = CrossAttention(config=config)
        self.coattention_avg_pool = nn.AvgPool1d(512)
        self.cross_avg_pool = nn.AvgPool1d(512)
        self.coattention_linear = nn.Linear(config.hidden_size, 1)
        self.cross_attention_linear = nn.Linear(config.hidden_size, 1)
        # self.co_act = gelu
        # self.cross_act = gelu
        # self.w1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # self.w2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # # initialization
        # self.w1.data.fill_(0.5)
        # self.w2.data.fill_(0.5)
        # self.w3.data.fill_(0.3)
        self.start_logits = nn.Linear(config.hidden_size, 1)
        self.end_logits = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sent = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                cls_weight=None,
                entity_ids=None,
                pq_end_pos=None,
                start_positions=None,
                end_positions=None,
                sent_mask=None,
                sent_lbs=None,
                sent_weight=None):
        if len(input_ids.shape) < 2:
            input_ids = input_ids.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            pq_end_pos = pq_end_pos.unsqueeze(0)
            if start_positions is not None and len(start_positions.shape) < 2:
                start_positions = start_positions.unsqueeze(0)
                end_positions = end_positions.unsqueeze(0)
                sent_mask = sent_mask.unsqueeze(0)
                sent_lbs = sent_lbs.unsqueeze(0)
                sent_weight = sent_weight.unsqueeze(0)
        # roberta 取消了NSP任务，所以token_type_ids没有作用
        outputs = self.electra(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        # add cross attention
        context_sequence_output, ques_sequence_output, context_attention_mask, ques_attention_mask = split_context_ques(
            sequence_output, pq_end_pos)
        cross_output = self.cross_attention(ques_sequence_output, sequence_output, ques_attention_mask)
        cross_cls_output = self.cross_avg_pool(cross_output.permute(0, 2, 1)).permute(0, 2, 1).squeeze(1)
        cross_cls_attention = self.cross_attention_linear(cross_cls_output).unsqueeze(1)
        # cross_cls_attention = self.cross_act(cross_cls_attention)


        # sequence_output = self.qa_net(C=context_sequence_output,
        #                               Q=ques_sequence_output,
        #                               cmask=context_attention_mask,
        #                               qmask=ques_attention_mask)
        ones_mask = torch.ones_like(attention_mask).cuda()
        context_mask = (ones_mask - token_type_ids) * attention_mask
        ques_mask = token_type_ids * attention_mask
        coattention_mask = torch.matmul(context_mask.unsqueeze(-1).float(), ques_mask.unsqueeze(1).float())
        sequence_output = self.coattention(sequence_output, sequence_output, coattention_mask)
        sequence_cls_output = self.coattention_avg_pool(sequence_output.permute(0, 2, 1)).permute(0, 2, 1).squeeze(1)
        coattention_cls_attention = self.coattention_linear(sequence_cls_output).unsqueeze(1)
        # coattention_cls_attention = self.co_act(coattention_cls_attention)

        # biattention
        # bi_attention_output, memory = self.bi_attention(context_sequence_output, ques_sequence_output,
        #                                                 ques_attention_mask)
        # bi_attention_output = self.bi_attn_linear(bi_attention_output)
        # bs, context_len, hidden_dim = bi_attention_output.shape
        # zeros = torch.zeros((bs, 512 - context_len, hidden_dim)).cuda()
        # bi_attention_output = torch.cat([bi_attention_output, zeros], dim=1)

        # bs, context_len, hidden_dim = sequence_output.shape
        # zeros = torch.zeros((bs, 512 - context_len, hidden_dim)).cuda()
        # sequence_output = torch.cat([sequence_output, zeros], dim=1)
        # sequence_output = self.w1 * sequence_output + self.w2 * cross_output + outputs[0] + self.w3 * bi_attention_output
        # sequence_output = self.w1 * sequence_output + self.w2 * cross_output + outputs[0]
        # sequence_output = 0.5 * sequence_output + 0.5 * cross_output + outputs[0]
        # sequence_output = sequence_output + outputs[0]
        sequence_output = cross_output * cross_cls_attention + sequence_output * coattention_cls_attention
        sequence_output = sequence_output + outputs[0]
        ones_mask = torch.ones_like(attention_mask).cuda()
        context_mask = (ones_mask - token_type_ids) * attention_mask
        extended_context_mask = (1.0 - context_mask) * -10000.0
        start_logits = self.start_logits(sequence_output).squeeze(-1) + extended_context_mask
        end_logits = self.end_logits(sequence_output).squeeze(-1) + extended_context_mask
        # 去除context mask
        sent_logits = self.sent(sequence_output).squeeze(-1) * context_mask.float()
        # sent_logits = self.sent(sequence_output).squeeze(-1)
        if len(sent_logits) > 1:
            sent_logits.squeeze(-1)
        loss_fn1 = torch.nn.BCEWithLogitsLoss(reduce=False, size_average=False)
        # 去除sent_mask
        # sent_logits = sent_logits * sent_mask.float()
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sent_lbs = sent_lbs[:, 0:context_maxlen]
            # sent_weight = sent_weight[:, 0:context_maxlen]
            sent_loss = loss_fn1(sent_logits, sent_lbs.float())
            # sent_loss = (sent_loss * sent_mask.float()) * sent_weight
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


class ElectraForQuestionAnsweringCrossAttentionCL(ElectraModel):
    def __init__(self, config):
        super(ElectraForQuestionAnsweringCrossAttentionCL, self).__init__(config)
        self.electra = ElectraModel(config)
        self.cross_attention = CrossAttention(config=config)
        self.start_logits = nn.Linear(config.hidden_size, 1)
        self.end_logits = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sent = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                cls_weight=None,
                entity_ids=None,
                pq_end_pos=None,
                start_positions=None,
                end_positions=None,
                sent_mask=None,
                sent_lbs=None,
                sent_weight=None):
        if len(input_ids.shape) < 2:
            input_ids = input_ids.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            pq_end_pos = pq_end_pos.unsqueeze(0)
            if start_positions is not None and len(start_positions.shape) < 2:
                start_positions = start_positions.unsqueeze(0)
                end_positions = end_positions.unsqueeze(0)
                sent_mask = sent_mask.unsqueeze(0)
                sent_lbs = sent_lbs.unsqueeze(0)
                sent_weight = sent_weight.unsqueeze(0)
        # roberta 取消了NSP任务，所以token_type_ids没有作用
        outputs = self.electra(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        ## add cross attention
        context_sequence_output, ques_sequence_output, context_attention_mask, ques_attention_mask = split_context_ques(
            sequence_output, pq_end_pos)
        sequence_output = self.cross_attention(ques_sequence_output, sequence_output, ques_attention_mask)
        sequence_output = sequence_output + outputs[0]
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
        # sent_logits = sent_logits * sent_mask.float()
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sent_lbs = sent_lbs[:, 0:context_maxlen]
            # sent_weight = sent_weight[:, 0:context_maxlen]
            sent_loss = loss_fn1(sent_logits, sent_lbs.float())
            # sent_loss = (sent_loss * sent_mask.float()) * sent_weight
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


class ElectraForQuestionAnsweringCrossAttentionWithDP(ElectraModel):
    def __init__(self, config):
        super(ElectraForQuestionAnsweringCrossAttentionWithDP, self).__init__(config)
        self.electra = ElectraModel(config)
        self.cross_attention = CrossAttention(config=config)
        self.start_logits = nn.Linear(config.hidden_size, 1)
        self.end_logits = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cross_drop = nn.Dropout(0.2)
        self.sent = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                cls_weight=None,
                entity_ids=None,
                pq_end_pos=None,
                start_positions=None,
                end_positions=None,
                sent_mask=None,
                sent_lbs=None,
                sent_weight=None):
        if len(input_ids.shape) < 2:
            input_ids = input_ids.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            pq_end_pos = pq_end_pos.unsqueeze(0)
            if start_positions is not None and len(start_positions.shape) < 2:
                start_positions = start_positions.unsqueeze(0)
                end_positions = end_positions.unsqueeze(0)
                sent_mask = sent_mask.unsqueeze(0)
                sent_lbs = sent_lbs.unsqueeze(0)
                sent_weight = sent_weight.unsqueeze(0)
        # roberta 取消了NSP任务，所以token_type_ids没有作用
        outputs = self.electra(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        ## add cross attention
        context_sequence_output, ques_sequence_output, context_attention_mask, ques_attention_mask = split_context_ques(
            sequence_output, pq_end_pos)
        sequence_output = self.cross_attention(ques_sequence_output, sequence_output, ques_attention_mask)
        sequence_output = sequence_output + outputs[0]
        sequence_output = self.cross_drop(sequence_output)
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
        # sent_logits = sent_logits * sent_mask.float()
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sent_lbs = sent_lbs[:, 0:context_maxlen]
            # sent_weight = sent_weight[:, 0:context_maxlen]
            sent_loss = loss_fn1(sent_logits, sent_lbs.float())
            # sent_loss = (sent_loss * sent_mask.float()) * sent_weight
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


class ElectraForQuestionAnsweringThreeCrossAttention(ElectraModel):
    def __init__(self, config):
        super(ElectraForQuestionAnsweringThreeCrossAttention, self).__init__(config)
        self.electra = ElectraModel(config)
        self.cross_attention1 = CrossAttention(config=config)
        self.cross_attention2 = CrossAttention(config=config)
        self.cross_attention3 = CrossAttention(config=config)
        self.start_logits = nn.Linear(config.hidden_size, 1)
        self.end_logits = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sent = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                cls_weight=None,
                entity_ids=None,
                pq_end_pos=None,
                start_positions=None,
                end_positions=None,
                sent_mask=None,
                sent_lbs=None,
                sent_weight=None):
        if len(input_ids.shape) < 2:
            input_ids = input_ids.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            pq_end_pos = pq_end_pos.unsqueeze(0)
            if start_positions is not None and len(start_positions.shape) < 2:
                start_positions = start_positions.unsqueeze(0)
                end_positions = end_positions.unsqueeze(0)
                sent_mask = sent_mask.unsqueeze(0)
                sent_lbs = sent_lbs.unsqueeze(0)
                sent_weight = sent_weight.unsqueeze(0)
        # roberta 取消了NSP任务，所以token_type_ids没有作用
        outputs = self.electra(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        ## add cross attention
        context_sequence_output, ques_sequence_output, context_attention_mask, ques_attention_mask = split_context_ques(
            sequence_output, pq_end_pos)
        sequence_output1 = self.cross_attention1(ques_sequence_output, sequence_output, ques_attention_mask)
        sequence_output2 = self.cross_attention2(ques_sequence_output, sequence_output, ques_attention_mask)
        sequence_output3 = self.cross_attention3(ques_sequence_output, sequence_output, ques_attention_mask)
        sequence_output1 = sequence_output1 + outputs[0]
        sequence_output2 = sequence_output2 + outputs[0]
        sequence_output3 = sequence_output3 + outputs[0]
        start_sequence_output = sequence_output1 + sequence_output2
        end_sequence_output = sequence_output1 + sequence_output3
        ones_mask = torch.ones_like(attention_mask).cuda()
        context_mask = (ones_mask - token_type_ids) * attention_mask
        extended_context_mask = (1.0 - context_mask) * -10000.0
        start_logits = self.start_logits(start_sequence_output).squeeze(-1) + extended_context_mask  # *context_mask.float()
        end_logits = self.end_logits(end_sequence_output).squeeze(-1) + extended_context_mask  # *context_mask.float()
        # 去除context mask
        sent_logits = self.sent(sequence_output).squeeze(-1) * context_mask.float()
        # sent_logits = self.sent(sequence_output).squeeze(-1)
        if len(sent_logits) > 1:
            sent_logits.squeeze(-1)
        loss_fn1 = torch.nn.BCEWithLogitsLoss(reduce=False, size_average=False)
        # 去除sent_mask
        # sent_logits = sent_logits * sent_mask.float()
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sent_lbs = sent_lbs[:, 0:context_maxlen]
            # sent_weight = sent_weight[:, 0:context_maxlen]
            sent_loss = loss_fn1(sent_logits, sent_lbs.float())
            # sent_loss = (sent_loss * sent_mask.float()) * sent_weight
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


class ElectraForQuestionAnsweringBiAttention(ElectraModel):
    def __init__(self, config):
        super(ElectraForQuestionAnsweringBiAttention, self).__init__(config)
        self.electra = ElectraModel(config)
        self.bi_attention = BiAttention(config.hidden_size, config.hidden_size, config.hidden_size)
        self.bi_attn_linear = nn.Linear(config.hidden_size * 4, config.hidden_size)
        self.bi_attn_dropout = nn.Dropout(0.2)
        self.start_logits = nn.Linear(config.hidden_size, 1)
        self.end_logits = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sent = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                cls_weight=None,
                entity_ids=None,
                pq_end_pos=None,
                start_positions=None,
                end_positions=None,
                sent_mask=None,
                sent_lbs=None,
                sent_weight=None):
        if len(input_ids.shape) < 2:
            input_ids = input_ids.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            pq_end_pos = pq_end_pos.unsqueeze(0)
            if start_positions is not None and len(start_positions.shape) < 2:
                start_positions = start_positions.unsqueeze(0)
                end_positions = end_positions.unsqueeze(0)
                sent_mask = sent_mask.unsqueeze(0)
                sent_lbs = sent_lbs.unsqueeze(0)
                sent_weight = sent_weight.unsqueeze(0)
        # roberta 取消了NSP任务，所以token_type_ids没有作用
        outputs = self.electra(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        ## add cross attention
        context_sequence_output, ques_sequence_output, context_attention_mask, ques_attention_mask = split_context_ques(
            sequence_output, pq_end_pos)

        bi_attention_output, memory = self.bi_attention(context_sequence_output, ques_sequence_output, ques_attention_mask)
        bi_attention_output = self.bi_attn_linear(bi_attention_output)
        bs, context_len, hidden_dim = bi_attention_output.shape
        zeros = torch.zeros((bs, 512 - context_len, hidden_dim)).cuda()
        bi_attention_output = torch.cat([bi_attention_output, zeros], dim=1)
        sequence_output = 0.5 * outputs[0] + 0.5 * bi_attention_output
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
        # sent_logits = sent_logits * sent_mask.float()
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sent_lbs = sent_lbs[:, 0:context_maxlen]
            # sent_weight = sent_weight[:, 0:context_maxlen]
            sent_loss = loss_fn1(sent_logits, sent_lbs.float())
            # sent_loss = (sent_loss * sent_mask.float()) * sent_weight
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


class ElectraForQuestionAnsweringCoAttention(ElectraModel):
    def __init__(self, config):
        super(ElectraForQuestionAnsweringCoAttention, self).__init__(config)
        self.electra = ElectraModel(config)
        self.co_attention = MyQANet(d_model=config.hidden_size)
        self.start_logits = nn.Linear(config.hidden_size, 1)
        self.end_logits = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sent = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                cls_weight=None,
                entity_ids=None,
                pq_end_pos=None,
                start_positions=None,
                end_positions=None,
                sent_mask=None,
                sent_lbs=None,
                sent_weight=None):
        if len(input_ids.shape) < 2:
            input_ids = input_ids.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            pq_end_pos = pq_end_pos.unsqueeze(0)
            if start_positions is not None and len(start_positions.shape) < 2:
                start_positions = start_positions.unsqueeze(0)
                end_positions = end_positions.unsqueeze(0)
                sent_mask = sent_mask.unsqueeze(0)
                sent_lbs = sent_lbs.unsqueeze(0)
                sent_weight = sent_weight.unsqueeze(0)
        # roberta 取消了NSP任务，所以token_type_ids没有作用
        outputs = self.electra(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        # add co attention
        context_sequence_output, ques_sequence_output, context_attention_mask, ques_attention_mask = split_context_ques(
            sequence_output, pq_end_pos)
        # cross_output = self.cross_attention(ques_sequence_output, sequence_output, ques_attention_mask)
        true_coattention = self.co_attention(C=context_sequence_output, Q=ques_sequence_output,
                                                 cmask=context_attention_mask, qmask=ques_attention_mask)
        bs, context_len, hidden_dim = true_coattention.shape
        zeros = torch.zeros((bs, 512 - context_len, hidden_dim)).cuda()
        true_coattention = torch.cat([true_coattention, zeros], dim=1)
        sequence_output = 0.5 * outputs[0] + 0.5 * true_coattention
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
        # sent_logits = sent_logits * sent_mask.float()
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sent_lbs = sent_lbs[:, 0:context_maxlen]
            # sent_weight = sent_weight[:, 0:context_maxlen]
            sent_loss = loss_fn1(sent_logits, sent_lbs.float())
            # sent_loss = (sent_loss * sent_mask.float()) * sent_weight
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


class ElectraForQuestionAnsweringSelfAttention(ElectraModel):
    def __init__(self, config):
        super(ElectraForQuestionAnsweringSelfAttention, self).__init__(config)
        self.electra = ElectraModel(config)
        self.transformer = TransformerLayer(hidden_size=config.hidden_size,
                                            head_num=config.num_attention_heads,
                                            dropout=config.hidden_dropout_prob,
                                            feedforward_size=config.hidden_size * 3)
        self.start_logits = nn.Linear(config.hidden_size, 1)
        self.end_logits = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sent = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                cls_weight=None,
                entity_ids=None,
                pq_end_pos=None,
                start_positions=None,
                end_positions=None,
                sent_mask=None,
                sent_lbs=None,
                sent_weight=None):
        if len(input_ids.shape) < 2:
            input_ids = input_ids.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            pq_end_pos = pq_end_pos.unsqueeze(0)
            if start_positions is not None and len(start_positions.shape) < 2:
                start_positions = start_positions.unsqueeze(0)
                end_positions = end_positions.unsqueeze(0)
                sent_mask = sent_mask.unsqueeze(0)
                sent_lbs = sent_lbs.unsqueeze(0)
                sent_weight = sent_weight.unsqueeze(0)
        # roberta 取消了NSP任务，所以token_type_ids没有作用
        outputs = self.electra(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        ## add self attention
        batch_size, seq_length, hidden_size = sequence_output.size()
        self_mask = attention_mask.unsqueeze(1).repeat(1, seq_length, 1).unsqueeze(1)
        self_attention_output = self.transformer(sequence_output, mask=self_mask)
        sequence_output = self_attention_output + outputs[0]
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
        # sent_logits = sent_logits * sent_mask.float()
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sent_lbs = sent_lbs[:, 0:context_maxlen]
            # sent_weight = sent_weight[:, 0:context_maxlen]
            sent_loss = loss_fn1(sent_logits, sent_lbs.float())
            # sent_loss = (sent_loss * sent_mask.float()) * sent_weight
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


class ElectraForQuestionAnsweringFFN(ElectraModel):
    def __init__(self, config):
        super(ElectraForQuestionAnsweringFFN, self).__init__(config)
        self.electra = ElectraModel(config)
        # self.transformer = TransformerLayer(hidden_size=config.hidden_size,
        #                                     head_num=24,
        #                                     dropout=config.hidden_dropout_prob,
        #                                     feedforward_size=config.hidden_size * 3)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 3),
            nn.ReLU(True),
            nn.Linear(config.hidden_size * 3, config.hidden_size),
            nn.ReLU(True)
        )
        self.start_logits = nn.Linear(config.hidden_size, 1)
        self.end_logits = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sent = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                cls_weight=None,
                entity_ids=None,
                pq_end_pos=None,
                start_positions=None,
                end_positions=None,
                sent_mask=None,
                sent_lbs=None,
                sent_weight=None):
        if len(input_ids.shape) < 2:
            input_ids = input_ids.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            pq_end_pos = pq_end_pos.unsqueeze(0)
            if start_positions is not None and len(start_positions.shape) < 2:
                start_positions = start_positions.unsqueeze(0)
                end_positions = end_positions.unsqueeze(0)
                sent_mask = sent_mask.unsqueeze(0)
                sent_lbs = sent_lbs.unsqueeze(0)
                sent_weight = sent_weight.unsqueeze(0)
        # roberta 取消了NSP任务，所以token_type_ids没有作用
        outputs = self.electra(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        ## add ffn output
        # batch_size, seq_length, hidden_size = sequence_output.size()
        # self_mask = attention_mask.unsqueeze(1).repeat(1, seq_length, 1).unsqueeze(1)
        # self_attention_output = self.transformer(sequence_output, mask=self_mask)
        ffn_output = self.ffn(sequence_output)
        sequence_output = ffn_output + outputs[0]
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
        # sent_logits = sent_logits * sent_mask.float()
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sent_lbs = sent_lbs[:, 0:context_maxlen]
            # sent_weight = sent_weight[:, 0:context_maxlen]
            sent_loss = loss_fn1(sent_logits, sent_lbs.float())
            # sent_loss = (sent_loss * sent_mask.float()) * sent_weight
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


class ElectraForQuestionAnsweringCrossAttentionOnReader(ElectraModel):
    def __init__(self, config):
        super(ElectraForQuestionAnsweringCrossAttentionOnReader, self).__init__(config)
        self.electra = ElectraModel(config)
        self.cross_attention = CrossAttention(config=config)
        self.start_logits = nn.Linear(config.hidden_size, 1)
        self.end_logits = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sent = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                cls_weight=None,
                entity_ids=None,
                pq_end_pos=None,
                start_positions=None,
                end_positions=None,
                sent_mask=None,
                sent_lbs=None,
                sent_weight=None):
        if len(input_ids.shape) < 2:
            input_ids = input_ids.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            pq_end_pos = pq_end_pos.unsqueeze(0)
            if start_positions is not None and len(start_positions.shape) < 2:
                start_positions = start_positions.unsqueeze(0)
                end_positions = end_positions.unsqueeze(0)
                sent_mask = sent_mask.unsqueeze(0)
                sent_lbs = sent_lbs.unsqueeze(0)
                sent_weight = sent_weight.unsqueeze(0)
        # roberta 取消了NSP任务，所以token_type_ids没有作用
        outputs = self.electra(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sent_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        sent_output = self.dropout(sent_output)
        ## add cross attention
        context_sequence_output, ques_sequence_output, context_attention_mask, ques_attention_mask = split_context_ques(
            sequence_output, pq_end_pos)
        sequence_output = self.cross_attention(ques_sequence_output, sequence_output, ques_attention_mask)
        sequence_output = sequence_output + outputs[0]
        ones_mask = torch.ones_like(attention_mask).cuda()
        context_mask = (ones_mask - token_type_ids) * attention_mask
        extended_context_mask = (1.0 - context_mask) * -10000.0
        start_logits = self.start_logits(sequence_output).squeeze(-1) + extended_context_mask  # *context_mask.float()
        end_logits = self.end_logits(sequence_output).squeeze(-1) + extended_context_mask  # *context_mask.float()
        # 去除context mask
        sent_logits = self.sent(sent_output).squeeze(-1) * context_mask.float()
        # sent_logits = self.sent(sequence_output).squeeze(-1)
        if len(sent_logits) > 1:
            sent_logits.squeeze(-1)
        loss_fn1 = torch.nn.BCEWithLogitsLoss(reduce=False, size_average=False)
        # 去除sent_mask
        # sent_logits = sent_logits * sent_mask.float()
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sent_lbs = sent_lbs[:, 0:context_maxlen]
            # sent_weight = sent_weight[:, 0:context_maxlen]
            sent_loss = loss_fn1(sent_logits, sent_lbs.float())
            # sent_loss = (sent_loss * sent_mask.float()) * sent_weight
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


class ElectraForQuestionAnsweringCrossAttentionOnSent(ElectraModel):
    def __init__(self, config):
        super(ElectraForQuestionAnsweringCrossAttentionOnSent, self).__init__(config)
        self.electra = ElectraModel(config)
        self.cross_attention = CrossAttention(config=config)
        self.start_logits = nn.Linear(config.hidden_size, 1)
        self.end_logits = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.reader_dropout = nn.Dropout(0.2)
        self.add_dropout = nn.Dropout(0.2)
        self.sent = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                cls_weight=None,
                entity_ids=None,
                pq_end_pos=None,
                start_positions=None,
                end_positions=None,
                sent_mask=None,
                sent_lbs=None,
                sent_weight=None):
        if len(input_ids.shape) < 2:
            input_ids = input_ids.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            pq_end_pos = pq_end_pos.unsqueeze(0)
            if start_positions is not None and len(start_positions.shape) < 2:
                start_positions = start_positions.unsqueeze(0)
                end_positions = end_positions.unsqueeze(0)
                sent_mask = sent_mask.unsqueeze(0)
                sent_lbs = sent_lbs.unsqueeze(0)
                sent_weight = sent_weight.unsqueeze(0)
        # roberta 取消了NSP任务，所以token_type_ids没有作用
        outputs = self.electra(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        reader_output = outputs[0]
        reader_output = self.reader_dropout(reader_output)
        sequence_output = self.dropout(sequence_output)
        ## add cross attention
        context_sequence_output, ques_sequence_output, context_attention_mask, ques_attention_mask = split_context_ques(
            sequence_output, pq_end_pos)
        sequence_output = self.cross_attention(ques_sequence_output, sequence_output, ques_attention_mask)
        sequence_output = sequence_output + outputs[0]
        sequence_output = self.add_dropout(sequence_output)
        ones_mask = torch.ones_like(attention_mask).cuda()
        context_mask = (ones_mask - token_type_ids) * attention_mask
        extended_context_mask = (1.0 - context_mask) * -10000.0
        start_logits = self.start_logits(reader_output).squeeze(-1) + extended_context_mask  # *context_mask.float()
        end_logits = self.end_logits(reader_output).squeeze(-1) + extended_context_mask  # *context_mask.float()
        # 去除context mask
        sent_logits = self.sent(sequence_output).squeeze(-1) * context_mask.float()
        # sent_logits = self.sent(sequence_output).squeeze(-1)
        if len(sent_logits) > 1:
            sent_logits.squeeze(-1)
        loss_fn1 = torch.nn.BCEWithLogitsLoss(reduce=False, size_average=False)
        # 去除sent_mask
        # sent_logits = sent_logits * sent_mask.float()
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sent_lbs = sent_lbs[:, 0:context_maxlen]
            # sent_weight = sent_weight[:, 0:context_maxlen]
            sent_loss = loss_fn1(sent_logits, sent_lbs.float())
            # sent_loss = (sent_loss * sent_mask.float()) * sent_weight
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


class ElectraForQuestionAnsweringForwardWithEntity(ElectraModel):
    def __init__(self, config):
        super(ElectraForQuestionAnsweringForwardWithEntity, self).__init__(config)
        self.electra = ElectraModel(config)
        ENTITY_NUM = 20
        ENTITY_DIM = 5
        self.entity_embedder = nn.Embedding(num_embeddings=ENTITY_NUM, embedding_dim=ENTITY_DIM)
        self.start_logits = nn.Linear(config.hidden_size + ENTITY_DIM, 1)
        self.end_logits = nn.Linear(config.hidden_size + ENTITY_DIM, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sent = nn.Linear(config.hidden_size + ENTITY_DIM, 1)
        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                entity_ids=None,
                cls_weight=None,
                start_positions=None,
                end_positions=None,
                sent_mask=None,
                sent_lbs=None,
                sent_weight=None):
        sequence_output = self.electra(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        sequence_output = self.dropout(sequence_output)
        entity_output = self.entity_embedder(entity_ids)
        sequence_output = torch.cat([sequence_output, entity_output], dim=-1)
        ones_mask = torch.ones_like(attention_mask).cuda()
        context_mask = (ones_mask - token_type_ids) * attention_mask
        extended_context_mask = (1.0 - context_mask) * -10000.0
        start_logits = self.start_logits(sequence_output).squeeze(-1) + extended_context_mask  # *context_mask.float()
        end_logits = self.end_logits(sequence_output).squeeze(-1) + extended_context_mask  # *context_mask.float()
        sent_logits = self.sent(sequence_output).squeeze(-1) * context_mask.float()
        # sent_logits = self.sent(sequence_output).squeeze(-1)
        if len(sent_logits) > 1:
            sent_logits.squeeze(-1)
        loss_fn1 = torch.nn.BCEWithLogitsLoss(reduce=False, size_average=False)
        # 去除sent_mask
        # sent_logits = sent_logits * sent_mask.float()
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sent_lbs = sent_lbs[:, 0:context_maxlen]
            # sent_weight = sent_weight[:, 0:context_maxlen]
            sent_loss = loss_fn1(sent_logits, sent_lbs.float())
            # sent_loss = (sent_loss * sent_mask.float()) * sent_weight
            # sent_loss = (sent_loss * sent_mask.float())
            sent_loss = torch.sum(sent_loss, (-1, -2), keepdim=False)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)  # bsz * seq bsz*n
            end_loss = loss_fct(end_logits, end_positions)
            ans_loss = start_loss + end_loss
            total_loss = ans_loss + 0.2 * sent_loss
            return total_loss, start_logits, end_logits, sent_logits
        else:
            start_logits = nn.Softmax(dim=-1)(start_logits)
            end_logits = nn.Softmax(dim=-1)(end_logits)
            sent_logits = torch.sigmoid(sent_logits)
            return start_logits, end_logits, sent_logits
