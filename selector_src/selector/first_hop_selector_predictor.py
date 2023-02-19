from __future__ import absolute_import, division, print_function
import os
import json

import argparse
import collections
import json
import logging
import math
import os
import random
import sys
from io import open
from pathlib import Path
import re
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, Sampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import BertTokenizer, RobertaTokenizer, ElectraTokenizer, AlbertTokenizer, AlbertConfig
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
from collections import Counter
import string
import gc


from first_hop_data_helper import (HotpotQAExample,
                                       HotpotInputFeatures,
                                       read_hotpotqa_examples,
                                       convert_examples_to_features)
from first_hop_selector import dev_feature_getter, write_predictions
from first_selector_predictor_config import get_config
sys.path.append("../pretrain_model")
from changed_model import BertForParagraphClassification, BertForRelatedSentence, \
    ElectraForParagraphClassification, ElectraForRelatedSentence, \
    RobertaForParagraphClassification, RobertaForRelatedSentence, \
    BertForParagraphClassificationMean, BertForParagraphClassificationMax, \
    ElectraForParagraphClassificationCrossAttention, AlbertForParagraphClassification
from optimization import BertAdam, warmup_linear

models_dict = {"BertForRelatedSentence": BertForRelatedSentence,
               "BertForParagraphClassification": BertForParagraphClassification,
               "BertForParagraphClassificationMean": BertForParagraphClassificationMean,
               "BertForParagraphClassificationMax": BertForParagraphClassificationMax,
               "ElectraForParagraphClassification": ElectraForParagraphClassification,
               "ElectraForRelatedSentence": ElectraForRelatedSentence,
               "RobertaForParagraphClassification": RobertaForParagraphClassification,
               "RobertaForRelatedSentence": RobertaForRelatedSentence,
               "ElectraForParagraphClassificationCrossAttention": ElectraForParagraphClassificationCrossAttention,
               "AlbertForParagraphClassification": AlbertForParagraphClassification
               }

# 日志设置
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def write_predict_result(examples, features, results, has_sentence_result=True):
    """ 给出预测结果 """
    tmp_best_paragraph = {}
    tmp_related_sentence = {}
    tmp_related_paragraph = {}
    paragraph_results = {}
    sentence_results = {}
    example_index2features = collections.defaultdict(list)
    for feature in features:
        example_index2features[feature.example_index].append(feature)
    unique_id2result = {}
    for result in results:
        unique_id2result[result[0]] = result
    for example_idx, example in enumerate(examples):
        features = example_index2features[example_idx]
        # 将5a8b57f25542995d1e6f1371_0_0 qid_context_sent 切分为 qid_context
        id = '_'.join(features[0].unique_id.split('_')[:-1])
        sentence_result = []
        if len(features) == 1:
            # 对单实例单结果处理
            get_feature = features[0]
            get_feature_id = get_feature.unique_id
            # 对max_seq预测的结果
            raw_result = unique_id2result[get_feature_id].logit
            # 第一个'[CLS]'为paragraph为支撑句标识
            paragraph_results[id] = raw_result[0]
            labels_result = raw_result
            cls_masks = get_feature.cls_mask
            if has_sentence_result:
                for idx, (label_result, cls_mask) in enumerate(zip(labels_result, cls_masks)):
                    if idx == 0:
                        continue
                    if cls_mask != 0:
                        sentence_result.append(label_result)
                sentence_results[id] = sentence_result
                assert len(sentence_result) == sum(features[0].cls_mask) - 1
        else:
            # 对单实例的多结果处理
            paragraph_result = 0
            overlap = 0
            mask1 = 0
            roll_back = None
            for feature_idx, feature in enumerate(features):
                feature_result = unique_id2result[feature.unique_id].logit
                if feature_result[0] > paragraph_result:
                    paragraph_result = feature_result[0]
                if has_sentence_result:
                    tmp_sent_result = []
                    tmp_label_result = []
                    mask1 += sum(feature.cls_mask[1:])
                    label_results = feature_result[1:]
                    cls_masks = feature.cls_mask[1:]
                    for idx, (label_result, cls_mask) in enumerate(zip(label_results, cls_masks)):
                        if cls_mask != 0:
                            tmp_sent_result.append(label_result)
                    if roll_back is None:
                        roll_back = 0
                    elif roll_back == 1:
                        sentence_result[-1] = max(sentence_result[-1], tmp_sent_result[0])
                        tmp_sent_result = tmp_sent_result[1:]
                    elif roll_back == 2:
                        sentence_result[-2] = max(sentence_result[-2], tmp_sent_result[0])
                        sentence_result[-1] = max(sentence_result[-1], tmp_sent_result[1])
                        tmp_sent_result = tmp_sent_result[2:]
                    sentence_result += tmp_sent_result
                    overlap += roll_back
                    roll_back = feature.roll_back
            paragraph_results[id] = paragraph_result
            sentence_results[id] = sentence_result
            if has_sentence_result:
                assert len(sentence_result) + overlap == mask1
    context_dict = {}
    # 将每个段落的结果写入到context中
    for k, v in paragraph_results.items():
        context_id, paragraph_id = k.split('_')
        # if context_id == '5a7b93905542997c3ec9722e':
        #     import pdb; pdb.set_trace()
        paragraph_id = int(paragraph_id)
        if context_id not in context_dict:
            context_dict[context_id] = [[-10000]*10, [-10000]*10]
        context_dict[context_id][0][paragraph_id] = v
    # 将context最大结果导出
    for k, v in context_dict.items():
        thread = 0.01
        tmp_related_paragraph[k] = v[0]
        max_v = max(v[0])
        min_v = min(v[0])
        max_logit = -1000
        max_result = False
        get_related_paras = []
        max_para = -1
        for idx, para_pro in enumerate(v[0]):
            if para_pro > max_logit:
                max_logit = para_pro
                max_para = idx
            para_pro = (para_pro - min_v)/ (max_v - min_v)
            if para_pro > thread:
                get_related_paras.append(idx)
        tmp_best_paragraph[k] = max_para
        # tmp_related_paragraph[k] = get_related_paras
    # 获取相关段落和句子
    if has_sentence_result:
        sentence_dict = {}
        for k, v in sentence_results.items():
            context_id, paragraph_id = k.split('_')
            paragraph_id = int(paragraph_id)
            if context_id not in sentence_dict:
                sentence_dict[context_id] = [[[]] * 10, [[]] * 10, [[]]*10]
            sentence_dict[context_id][0][paragraph_id] = v
        for k, v in sentence_dict.items():
            get_paragraph_idx = tmp_best_paragraph[k]
            pred_sent_result = v[0][get_paragraph_idx]
            real_sent_result = v[1][get_paragraph_idx]
            tmp_related_sentence[k] = [pred_sent_result, real_sent_result]
    return tmp_best_paragraph, tmp_related_sentence, tmp_related_paragraph


def run_predict(args):
    """ 预测结果 """
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    # 设置tokenizer
    cls_token = '[CLS]'
    sep_token = '[SEP]'
    unk_token = '[UNK]'
    pad_token = '[PAD]'
    if 'electra' in args.bert_model.lower():
        if not args.no_network:
            tokenizer = ElectraTokenizer.from_pretrained(args.bert_model,
                                                         do_lower_case=args.do_lower_case)
        else:
            tokenizer = ElectraTokenizer.from_pretrained(args.checkpoint_path,
                                                         do_lower_case=args.do_lower_case)
    elif 'albert' in args.bert_model.lower():
        cls_token = '[CLS]'
        sep_token = '[SEP]'
        pad_token = '<pad>'
        unk_token = '<unk>'
        config_class = AlbertConfig()
        config = config_class.from_pretrained(args.bert_model)
        if not args.no_network:
            tokenizer = AlbertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
        else:
            tokenizer = AlbertTokenizer.from_pretrained(args.checkpoint_path, do_lower_case=args.do_lower_case)

    elif 'roberta' in args.bert_model.lower():
        cls_token = '<s>'
        sep_token = '</s>'
        unk_token = '<unk>'
        pad_token = '<pad>'
        if not args.no_network:
            tokenizer = RobertaTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
        else:
            tokenizer = RobertaTokenizer.from_pretrained(args.checkpoint_path, do_lower_case=args.do_lower_case)
    elif 'bert' in args.bert_model.lower():
        if not args.no_network:
            tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
        else:
            tokenizer = BertTokenizer.from_pretrained(args.checkpoint_path, do_lower_case=args.do_lower_case)
    else:
        raise ValueError("Not implement!")

    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=args.do_lower_case)
    # 从文件中加载模型
    model = models_dict[args.model_name].from_pretrained(args.checkpoint_path)

    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
    dev_examples = read_hotpotqa_examples(input_file=args.dev_file,
                                          is_training='test')
    example_num = len(dev_examples)
    logger.info("all examples number: {}".format(example_num))
    max_train_data_size = 100000
    start_idxs = list(range(0, example_num, max_train_data_size))
    end_idxs = [x + max_train_data_size for x in start_idxs]
    end_idxs[-1] = example_num
    best_paragraph = {}
    related_sentence = {}
    related_paragraph = {}
    # new_context = {}
    total = 0
    max_len = 0

    has_sentence_result = True

    if args.model_name == 'BertForParagraphClassification' or 'ParagraphClassification' in args.model_name:
        has_sentence_result = False

    for idx in range(len(start_idxs)):
        logger.info("predict idx: {} all length: {}".format(idx, len(start_idxs)))
        truly_examples = dev_examples[start_idxs[idx]: end_idxs[idx]]
        truly_features = convert_examples_to_features(
            examples=truly_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            is_training='test',
            cls_token=cls_token,
            sep_token=sep_token,
            unk_token=unk_token,
            pad_token=pad_token
        )
        logger.info("all truly gotten features: {}".format(len(truly_features)))
        d_all_input_ids = torch.tensor([f.input_ids for f in truly_features], dtype=torch.long)
        d_all_input_mask = torch.tensor([f.input_mask for f in truly_features], dtype=torch.long)
        d_all_segment_ids = torch.tensor([f.segment_ids for f in truly_features], dtype=torch.long)
        d_all_cls_mask = torch.tensor([f.cls_mask for f in truly_features], dtype=torch.long)
        d_all_pq_end_pos = torch.tensor([f.pq_end_pos for f in truly_features], dtype=torch.long)
        d_all_cls_weight = torch.tensor([f.cls_weight for f in truly_features], dtype=torch.float)
        d_all_example_index = torch.arange(d_all_input_ids.size(0), dtype=torch.long)
        dev_data = TensorDataset(d_all_input_ids, d_all_input_mask, d_all_segment_ids,
                                 d_all_cls_mask, d_all_pq_end_pos, d_all_cls_weight, d_all_example_index)
        if args.local_rank == -1:
            dev_sampler = SequentialSampler(dev_data)
        else:
            dev_sampler = DistributedSampler(dev_data)
        dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.val_batch_size)
        RawResult = collections.namedtuple("RawResult",
                                           ["unique_id", "logit"])

        model.eval()
        with torch.no_grad():
            cur_result = []
            for idx, batch in enumerate(tqdm(dev_dataloader, desc="predict interation: {}".format(args.dev_file))):
                # example index getter
                d_example_indices = batch[-1]
                # 多gpu训练的scatter
                if n_gpu == 1:
                    # 去除example index
                    batch = tuple(x.squeeze(0).to(device) for x in batch[:-1])
                else:
                    batch = batch[:-1]
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "cls_mask": batch[3],
                    "pq_end_pos": batch[4],
                    "cls_weight": batch[5]
                }
                # 获取预测结果
                dev_logits = model(**inputs)
                dev_logits = torch.sigmoid(dev_logits)
                for i, example_index in enumerate(d_example_indices):
                    # start_position = start_positions[i].detach().cpu().tolist()
                    # end_position = end_positions[i].detach().cpu().tolist()
                    if not has_sentence_result:
                        dev_logit = dev_logits[i].detach().cpu().tolist()
                        dev_logit.reverse()
                    else:
                        dev_logit = dev_logits[i].detach().cpu().tolist()
                    dev_feature = truly_features[example_index.item()]
                    unique_id = dev_feature.unique_id
                    cur_result.append(RawResult(unique_id=unique_id,
                                             logit=dev_logit))

            tmp_best_paragraph, tmp_related_sentence, tmp_related_paragraph = write_predict_result(examples=truly_examples,
                                                                                                   features=truly_features,
                                                                                                   results=cur_result,
                                                                                                   has_sentence_result=has_sentence_result)
            best_paragraph.update(tmp_best_paragraph)
            related_sentence.update(tmp_related_sentence)
            related_paragraph.update(tmp_related_paragraph)
            del tmp_best_paragraph, tmp_related_sentence, tmp_related_paragraph
            del d_example_indices, inputs
            del cur_result
            gc.collect()
        del d_all_input_ids, d_all_input_mask, d_all_segment_ids, d_all_cls_mask, d_all_cls_weight, d_all_pq_end_pos, d_all_example_index
        del truly_examples, truly_features, dev_data, dev_logits
        gc.collect()
    # 获取新的文档
    logger.info("start saving data...")
    logger.info("writing result to file...")
    if not os.path.exists(args.predict_result_path):
        logger.info("make new output dir:{}".format(args.predict_result_path))
        os.makedirs(args.predict_result_path)
    json.dump(best_paragraph, open("{}/{}".format(args.predict_result_path, args.best_paragraph_file), "w", encoding="utf-8"))
    # json.dump(new_context, open("{}/{}".format(args.predict_result_path, args.new_context_file),
    #                             "w", encoding="utf-8"))
    json.dump(related_paragraph, open("{}/{}".format(args.predict_result_path, args.related_paragraph_file), "w", encoding='utf-8'))
    logger.info("write result done!")


if __name__ == '__main__':
    parser = get_config()
    args = parser.parse_args()
    run_predict(args=args)




