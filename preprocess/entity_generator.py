import argparse
import json
import os
from tqdm import tqdm
from multiprocessing import Pool

import spacy

nlp = spacy.load("en_core_web_sm")


def entity_process(data):
    data['entity_info'] = []
    question = data['question']
    question_entity_labels = ['' for _ in range(len(question))]
    question_doc = nlp(question)
    for ent in question_doc.ents:
        range_num = ent.end_char - ent.start_char
        ent_label = ent.label_
        question_entity_labels[ent.start_char: ent.end_char] = [ent_label for _ in range(range_num)]
    data['question_entity_info'] = question_entity_labels
    for ind_con, context in enumerate(data['context']):
        sents_entity_labels = []
        sents = context[1]
        for sent in sents:
            doc = nlp(sent)
            sent_entity_labels = ['' for _ in range(len(sent))]
            for ent in doc.ents:
                range_num = ent.end_char - ent.start_char
                ent_label = ent.label_
                sent_entity_labels[ent.start_char: ent.end_char] = [ent.label_ for _ in range(range_num)]
            sents_entity_labels.append(sent_entity_labels)
        data['entity_info'].append(sents_entity_labels)
    return data


def generate_entity_label(input_file,
                          output_file,
                          dict_file):
    """"""
    datas = None
    with open(input_file, "r") as f:
        datas = json.load(f)
    pool = Pool(10)
    new_datas = []
    for result in tqdm(pool.imap(func=entity_process, iterable=datas),
                       total=len(datas),
                       desc="getting word entity info..."):
        new_datas.append(result)
    pool.close()
    pool.join()
    json.dump(new_datas, open(output_file, 'w', encoding='utf-8'))


def generate_entity_label_single_process(input_file,
                                         output_file,
                                         dict_file):
    with open(input_file, "r") as f:
        datas = json.load(f)
        datas = datas[:1000]
    all_ner_info_dict = {}
    for data in tqdm(datas):
        data['entity_info'] = []
        # TODO 增加query的实体识别
        for ind_con, context in enumerate(data['context']):
            sents_entity_labels = []
            sents = context[1]
            for sent in sents:
                doc = nlp(sent)
                sent_entity_labels = ['' for _ in range(len(sent))]
                for ent in doc.ents:
                    range_num = ent.end_char - ent.start_char
                    ent_label = ent.label_
                    if ent_label not in all_ner_info_dict:
                        all_ner_info_dict[ent_label] = 1
                    else:
                        all_ner_info_dict[ent_label] += 1
                    sent_entity_labels[ent.start_char: ent.end_char] = [ent.label_ for _ in range(range_num)]
                sents_entity_labels.append(sent_entity_labels)
            data['entity_info'].append(sents_entity_labels)
    print(all_ner_info_dict)


if __name__ == '__main__':
    input_files = ["../../data/hotpot_data/hotpot_train_labeled_data_v3.json",
                    "../../data/hotpot_data/hotpot_dev_labeled_data_v3.json"]
    output_files = ["../../data/hotpot_data/hotpot_train_labeled_data_v3_with_entity_label.json",
                    "../../data/hotpot_data/hotpot_dev_labeled_data_v3_with_entity_label.json"]
    dict_files = ["../../data/hotpot_data/hotpot_train_labeled_data_v3_entity_dict.json",
                  "../../data/hotpot_data/hotpot_dev_labeled_data_v3_entity_dict.json"
                  ]
    for input_file, output_file, dict_file in zip(input_files, output_files, dict_files):
        generate_entity_label(input_file=input_file,
                              output_file=output_file,
                              dict_file=dict_file)
