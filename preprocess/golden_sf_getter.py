import json
from tqdm import tqdm


def get_golden_paragraph(input_file, output_file):
    """ 获取真实的答案支撑句 """
    data = json.load(open(input_file, "r"))
    sp_dict = {}
    for info in tqdm(data):
        sup = info['supporting_facts']
        contexts = info['context']
        get_id = info['_id']
        sp_dict[get_id] = []
        for context_idx, context in enumerate(contexts):
            title, sentences = context
            for sentence_idx, sentence in enumerate(sentences):
                if [title, sentence_idx] in sup:
                    if get_id in sp_dict:
                        sp_dict[get_id].append(context_idx)
        sp_dict[get_id] = list(set(sp_dict[get_id]))
    print(output_file)
    json.dump(sp_dict, open(output_file, "w"))


if __name__ == '__main__':
    input_files = ["../../data/hotpot_data/hotpot_train_labeled_data_v3.json",
                   "../../data/hotpot_data/hotpot_dev_labeled_data_v3.json"]
    output_files = ["../../data/hotpot_data/train_golden.json",
                    "../../data/hotpot_data/dev_golden.json"]
    for input_file, output_file in zip(input_files, output_files):
        get_golden_paragraph(input_file, output_file)

