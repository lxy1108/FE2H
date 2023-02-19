import os
import collections
import json

def prediction_evaluate_origin(args,
                        paragraph_results,
                        labels,
                        thread=0.5):
    """ 对预测进行评估 """
    p_recall = p_precision = sent_em = sent_acc = sent_recall = 0
    all_count = 0
    new_para_result = {}
    for k, v in paragraph_results.items():
        q_id, context_id = k.split('_')
        context_id = int(context_id)
        if q_id not in new_para_result:
            new_para_result[q_id] = [[0] * 10, [0] * 10]
        new_para_result[q_id][0][context_id] = v
    for k, v in labels.items():
        q_id, context_id = k.split('_')
        context_id = int(context_id)
        new_para_result[q_id][1][context_id] = v[0]
    for k, v in new_para_result.items():
        all_count += 1
        p11 = p10 = p01 = p00 = 0
        max_v = max(v[0])
        min_v = min(v[0])
        max_logit = -100
        max_result = False
        # TODO: check v format
        for idx, (paragraph_result, label) in enumerate(zip(v[0], v[1])):
            if paragraph_result > max_logit:
                max_logit = paragraph_result
                max_result = True if label == 1 else max_result
            # MinMax Scaling
            paragraph_result = (paragraph_result - min_v) / (max_v - min_v)
            paragraph_result = 1 if paragraph_result > thread else 0
            if paragraph_result == 1 and label == 1:
                p11 += 1
            elif paragraph_result == 1 and label == 0:
                p10 += 1
            elif paragraph_result == 0 and label == 1:
                p01 += 1
            elif paragraph_result == 0 and label == 0:
                p00 += 1
            else:
                # TODO: check the function
                raise NotImplemented
        if p11 + p01 != 0:
            p_recall += p11 / (p11 + p01)
        else:
            print("error in calculate paragraph recall!")
        if p11 + p10 != 0:
            p_precision += p11 / (p11 + p10)
        else:
            print("error in calculate paragraph precision!")
        if p11 == 2 and p10 == 0:
            sent_em += 1
        if p01 == 0:
            sent_recall += 1
        if max_result:
            sent_acc += 1
    return sent_acc / all_count, p_precision / all_count, sent_em / all_count, sent_recall / all_count


def prediction_evaluate(args,
                        paragraph_results,
                        labels,
                        thread=0.5,
                        step=0):
    """ 对预测进行评估 """
    p_recall = p_precision = sent_em = sent_acc = sent_recall = 0
    all_count = 0
    new_para_result = {}
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    print("predict paragraph result length: {}".format(len(paragraph_results)))
    with open(os.path.join(args.output_dir, "predict_second_dev_paragraph_result_{}.json".format(step)), "w") as writer:
        json.dump(paragraph_results, writer)
    for k, v in paragraph_results.items():
        q_id, context_id = k.split('_')
        context_id = int(context_id)
        if q_id not in new_para_result:
            new_para_result[q_id] = [-100000] * 10
        new_para_result[q_id][context_id] = v
    with open(args.dev_file, "r") as f:
        dev_data = json.load(f)
    first_best_paragraph_file = "{}/{}".format(args.first_predict_result_path, args.dev_best_paragraph_file)
    first_best_paragraph = json.load(open(first_best_paragraph_file, 'r', encoding='utf-8'))
    # 读取验证集数据
    dev_dict = {}
    for info in dev_data:
        get_id = info['_id']
        sfs = info['supporting_facts']
        paragraphs = info['context']
        for context_idx, context in enumerate(paragraphs):
            title, sentences = context
            for sent_idx, sent in enumerate(sentences):
                if [title, sent_idx] in sfs:
                    if get_id not in dev_dict:
                        dev_dict[get_id] = []
                    dev_dict[get_id].append(context_idx)
                    dev_dict[get_id] = list(set(dev_dict[get_id]))
    predict_true_num = 0
    em_num = 0
    predict_dict = {}
    for q_id, info in dev_dict.items():
        first_paragraph = first_best_paragraph[q_id]
        predict_info = new_para_result[q_id]
        predict_info[first_paragraph] = -10000
        max_value = max(predict_info)
        second_paragraph = 0
        for predict_idx, predict_val in enumerate(predict_info):
            if predict_val == max_value:
                second_paragraph = predict_idx
                break
        all_count += len(info)
        if first_paragraph in info:
            predict_true_num += 1
        if second_paragraph in info:
            predict_true_num += 1
        if first_paragraph in info and second_paragraph in info:
            em_num += 1
        predict_dict[q_id] = [first_paragraph, second_paragraph]

    with open(os.path.join(args.output_dir, "predict_second_all_dev_result_{}.json".format(step)), "w") as writer:
        json.dump(new_para_result, writer)
    with open(os.path.join(args.output_dir, "predict_second_dev_result_{}.json".format(step)), "w") as writer:
        json.dump(predict_dict, writer)
    recall = 1.0 * predict_true_num / all_count
    precision = 1.0 * predict_true_num / (len(new_para_result) * 2)
    f1 = 2.0 * (recall * precision) / (recall + precision)
    em = 1.0 * em_num / len(dev_dict)
    return recall, precision, f1, em


def write_predictions(args, all_examples, all_features, all_results, is_training='train', has_sentence_result=True, step=0):
    """ 将预测结果写入json文件 """
    example_index2features = collections.defaultdict(list)
    for feature in all_features:
        example_index2features[feature.example_index].append(feature)
    unique_id2result = {x[0]: x for x in all_results}
    paragraph_results = {}
    sentence_results = {}
    labels = {}
    for example_index, example in enumerate(all_examples):
        features = example_index2features[example_index]
        # TODO: check id格式
        # 将5a8b57f25542995d1e6f1371_0_0 qid_context_sent 切分为 qid_context
        get_id = '_'.join(features[0].unique_id.split('_')[:-1])
        sentence_result = []
        sentence_all_labels = []
        if len(features) == 1:
            # 对单实例的单结果处理
            get_feature = features[0]
            get_feature_id = get_feature.unique_id
            # 对max_seq预测的结果
            raw_result = unique_id2result[get_feature_id].logit
            # 第一个'[CLS]'为paragraph为支撑句标识
            paragraph_results[get_id] = raw_result[0]
            labels_result = raw_result
            cls_masks = get_feature.cls_mask
            cls_labels = get_feature.cls_label
            if has_sentence_result:
                for idx, (label_result, cls_mask, cls_label) in enumerate(zip(labels_result, cls_masks, cls_labels)):
                    if idx == 0:
                        sentence_all_labels.append(cls_label)
                        continue
                    if cls_mask != 0:
                        sentence_all_labels.append(cls_label)
                        sentence_result.append(label_result)
                sentence_results[get_id] = sentence_result
                assert len(sentence_result) == sum(features[0].cls_mask) - 1
                assert len(sentence_all_labels) == sum(features[0].cls_mask)
            else:
                sentence_all_labels.append(cls_labels[0])
            labels[get_id] = sentence_all_labels
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
                tmp_sent_result = []
                tmp_label_result = []
                mask1 += sum(feature.cls_mask[1:])
                label_results = feature_result[1:]
                cls_masks = feature.cls_mask[1:]
                cls_labels = feature.cls_label[1:]
                if has_sentence_result:
                    for idx, (label_result, cls_mask, cls_label) in enumerate(zip(label_results, cls_masks, cls_labels)):
                        if cls_mask != 0:
                            # TODO: check the order of append
                            tmp_label_result.append(cls_label)
                            tmp_sent_result.append(label_result)
                    if roll_back is None:
                        roll_back = 0
                        sentence_all_labels.append(feature.cls_label[0])
                        sentence_all_labels += tmp_label_result
                    elif roll_back == 1:
                        sentence_result[-1] = max(sentence_result[-1], tmp_sent_result[0])
                        tmp_sent_result = tmp_sent_result[1:]
                        if sentence_all_labels[0] == 0 and feature.cls_label[0] == 1:
                            sentence_all_labels[0] = 1
                        sentence_all_labels += tmp_label_result[1:]
                    elif roll_back == 2:
                        sentence_result[-2] = max(sentence_result[-2], tmp_sent_result[0])
                        sentence_result[-1] = max(sentence_result[-1], tmp_sent_result[1])
                        tmp_sent_result = tmp_sent_result[2:]
                        if sentence_all_labels[0] == 0 and feature.cls_label[0] == 1:
                            sentence_all_labels[0] = 1
                        sentence_all_labels += tmp_label_result[2:]
                    else:
                        sentence_all_labels += tmp_label_result
                    sentence_result += tmp_sent_result
                    overlap += roll_back
                    roll_back = feature.roll_back
                else:
                    if len(sentence_all_labels) == 0:
                        sentence_all_labels.append(cls_labels[0])
                    else:
                        sentence_all_labels[0] = max(sentence_all_labels[0], cls_labels[0])
            paragraph_results[get_id] = paragraph_result
            sentence_results[get_id] = sentence_result
            labels[get_id] = sentence_all_labels
            if has_sentence_result:
                assert len(sentence_result) + overlap == mask1
                assert len(sentence_all_labels) + overlap == mask1 + 1
    # return paragraph_results, sentence_results, labels
    if is_training == 'test':
        return 0, 0, 0, 0
    else:
        return prediction_evaluate(args,
                                   paragraph_results=paragraph_results,
                                   labels=labels,
                                   step=step)