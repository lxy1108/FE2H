import json


def naive_preprocess(config):
    """ 预处理得到支撑段落中最开始的那个句子 """
    input_data = json.load(open(config.input_file, 'r'))
    all_no_answer_num = 0
    pre_process_data = []
    for qa_info in input_data:
        # 使用支撑句里的答案作为标准答案
        supporting_facts = qa_info['supporting_facts']
        title2sf = {}
        for supporting_fact in supporting_facts:
            # 将支撑句按照title+list(sent_idx)方式写入字典里
            get_title = supporting_fact[0]
            if supporting_fact[0] not in title2sf:
                title2sf[get_title] = []
            title2sf[get_title].append(supporting_fact[1])
        title2idx = {context[0]: idx for idx, context in enumerate(qa_info['context'])}
        has_answer = False
        for get_title, sf_sent_idxs in title2sf.items():
            if has_answer:
                continue
            sf_sent_idxs = sorted(sf_sent_idxs)
            context_idx = title2idx[get_title]
            para = ''.join(qa_info['context'][context_idx][1])
            answers = []
            has_answer = True
            if qa_info['answer'].lower() == 'yes':
                answers.append([-1, -1, -1])
                qa_info['label'] = answers
            elif qa_info['answer'].lower() == 'no':
                answers.append([-2, -2, -2])
                qa_info['labels'] = answers
            elif qa_info['answer'] in para:
                get_answer_idx = para.find(qa_info['answer'])
                answers.append([get_title, get_answer_idx, len(qa_info['answer']) + get_answer_idx])
            else:
                has_answer = False
            if has_answer:
                qa_info['labels'] = answers
                pre_process_data.append(qa_info)
        if not has_answer:
            all_no_answer_num += 1
            print('qid:{} answer:{} has no answer!'.format(qa_info['_id'], qa_info['answer']))
    print('all_no_answer_num: {} and get {} preprocessed data'.format(all_no_answer_num, len(pre_process_data)))
    json.dump(pre_process_data, open(config.preprocessed_file, 'w', encoding='utf-8'))

