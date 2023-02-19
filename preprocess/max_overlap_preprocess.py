import re
import json

from preprocess_util import word_tokenize, convert_idx, fix_span, delete_stopwords


def max_overlap_preprocess(config, windows_size=10):
    """ 得到与问题重合度最大的篇章里的索引即得到答案的确切位置 重合度由答案附近10个词决定"""
    input_data = json.load(open(config.input_file, 'r'))
    all_no_answer_num = 0
    different_num = 0
    pre_process_data = []
    for qa_info in input_data:
        supporting_facts = qa_info['supporting_facts']
        sentence_offsets = []
        context_idx2para_idx = {} # 包含支撑句的索引转化为段落索引
        context_text = ""
        title2sentences = {}
        answers = []
        if qa_info['answer'].lower() == 'yes':
            answers.append([-1, -1, -1])
            qa_info['labels'] = answers
            pre_process_data.append(qa_info)
            continue
        if qa_info['answer'].lower() == 'no':
            answers.append([-2, -2, -2])
            qa_info['labels'] = answers
            pre_process_data.append(qa_info)
            continue
        # 处理答案不在支撑句中的情况或者答案错误，如问题5a7a870d55429941d65f2688 答案为Portugal. The Ma实际应该为'Portugal.',
        #     ' The Man is an American rock band from Wasilla, Alaska.'
        if qa_info['_id'] == '5a7e43155542991319bc9457':
            supporting_facts.append(['Ivan Allen Jr. Prize for Social Courage', 0])
        if qa_info['_id'] == '5ac34c7a554299741d48a245':
            supporting_facts.append(['Medusa (Supergirl)', 2])
        if qa_info['_id'] == '5a74a82f5542996c70cfada3':
            supporting_facts.append(['Love the Way You Lie (Part II)', 4])
        if qa_info['_id'] == '5ae0e421554299422ee99543':
            supporting_facts.append(["Billy Budd (opera)", 0])
        if qa_info['_id'] == '5abfedab5542997d6429594b':
            supporting_facts.append(["Am\u00e9lie Mauresmo", 1])
        if qa_info['_id'] == '5abc54525542996583600444':
            supporting_facts.append(['Mina and the Count', 1])
            supporting_facts.append(['Oh Yeah! Cartoons', 0])
        if qa_info['_id'] == '5a87255c5542996432c5722f':
            supporting_facts.append(["Paul Jr. Designs", 0])
        if qa_info['_id'] == '5a7a525055429941d65f25ae':
            supporting_facts.append(["Tora! Tora! Tora!", 0])
            supporting_facts.append(["Tora! Tora! Tora!", 1])
            supporting_facts.append(["Tora! Tora! Tora!", 2])
        if qa_info['_id'] == '5ab55d725542992aa134a2cf':
            supporting_facts.append(["Bailout! The Game", 0])
        if qa_info['_id'] == '5a77218b55429937353601e2':
            supporting_facts.append(["Dynamite!! 2010", 0])
            supporting_facts.append(["Dynamite!! 2010", 1])
            supporting_facts.append(["Dynamite!! 2010", 2])
            supporting_facts.append(["Dynamite!! 2010", 3])
        if qa_info['_id'] == '5a8f3aac55429918e830d1db':
            supporting_facts.append(["Harry Potter and the Prisoner of Azkaban (film)", 0])
        if qa_info['_id'] == '5a746d6555429979e288294d':
            supporting_facts.append(["Anvil! The Story of Anvil", 0])
            supporting_facts.append(["Anvil! The Story of Anvil", 1])
        if qa_info['_id'] == '5a7392f755429905862fe063':
            supporting_facts.append(["Scooby-Doo! and the Samurai Sword", 0])
        if qa_info['_id'] == '5a7939fd55429970f5fffe7e':
            supporting_facts.append(["Jun. K", 1])
        if qa_info['_id'] == '5abe38ec5542993f32c2a099':
            supporting_facts.append(["Godspeed You! Black Emperor", 1])
        if qa_info['_id'] == '5abbf2475542993f40c73c25':
            supporting_facts.append(["Portugal. The Man", 1])
        if qa_info['_id'] == '5abaa6cb55429901930fa879':
            supporting_facts.append(["Tiger! Tiger! (Kipling short story)", 1])
            supporting_facts.append(["Tiger! Tiger! (Kipling short story)", 2])
            supporting_facts.append(["Letting in the Jungle", 0])
            supporting_facts.append(["Letting in the Jungle", 2])
        if qa_info['_id'] == '5a7a870d55429941d65f2688':
            supporting_facts.append(["Portugal. The Man", 0])
            supporting_facts.append(["Portugal. The Man", 1])
        if qa_info['_id'] == '5a7555215542996c70cfaee1':
            supporting_facts.append(["Hey Pa! There's a Goat on the Roof", 0])
        if qa_info['_id'] == '5abdd08d5542991f6610604c':
            supporting_facts.append(['Guardians of the Galaxy Vol. 2', 0])
            supporting_facts.append(['Guardians of the Galaxy Vol. 2', 1])
            supporting_facts.append(['Guardians of the Galaxy Vol. 2', 2])
        if qa_info['_id'] == '5adde6535542990dbb2f7ef9':
            supporting_facts.append(["Hail! Hail! Rock 'n' Roll", 0])
            supporting_facts.append(["Hail! Hail! Rock 'n' Roll", 1])
        for context_idx, context_info in enumerate(qa_info["context"]):
            pre_sentence_idx = -1
            paragraph_text = ""
            title, sentences = context_info
            for sentence_idx, sentence in enumerate(sentences):
                if [title, sentence_idx] not in supporting_facts:
                    paragraph_text += sentence
                    continue
                # 答案一定在支撑句中，若答案跨句子则答案也一定在多个支撑句拼接中
                if title not in title2sentences:
                    title2sentences[title] = [sentence, ]
                    context_idx2para_idx[title] = {}
                else:
                    if pre_sentence_idx + 1 == sentence_idx:
                        title2sentences[title][-1] += sentence
                    else:
                        title2sentences[title].append(sentence)
                pre_sentence_idx = sentence_idx
                for ch_idx, ch in enumerate(sentence):
                    context_idx2para_idx[title][len(context_text) + ch_idx] = len(paragraph_text) + ch_idx
                sentence_tokens = word_tokenize(sentence)
                sentence_spans = convert_idx(sentence, sentence_tokens)
                cur_context_length = len(context_text)
                # 转化为全局span
                sentence_spans = [(cur_context_length + s[0], cur_context_length + s[1]) for s in sentence_spans]
                sentence_offsets.append(sentence_spans)
                context_text += sentence
                paragraph_text += sentence
        # 答案在支撑句中
        if qa_info['answer'] not in context_text:
            qa_info['answer'] = qa_info['answer'][:-1]
        if qa_info['answer'] not in context_text:
            print(qa_info['answer'])
            print(qa_info['_id'])

        assert qa_info['answer'] in context_text, "answer not in support facts: qas_id:{} answer: {}".format(
            qa_info['_id'], qa_info['answer'])
        question_tokens = word_tokenize(qa_info['question'])
        all_overlaps = []
        for title, sentences in title2sentences.items():
            sentence_overlap = []
            # 为拼接后的sentence
            for sentence_idx, sentence in enumerate(sentences):
                sentence_tokens = word_tokenize(sentence)
                sentence_spans = convert_idx(sentence, sentence_tokens)
                # 寻找答案
                for m in re.finditer(re.escape(qa_info['answer']), sentence):
                    start_token = None
                    end_token = None
                    for span_idx, span in enumerate(sentence_spans):
                        if span[0] <= m.span()[0] < span[1]:
                            start_token = span_idx
                        if span[0] <= m.span()[1] <= span[1]:
                            end_token = span_idx
                    assert start_token is not None and end_token is not None
                    # 答案上下10个单词与question的重合度
                    if len(sentence_tokens) <= windows_size + (end_token - start_token + 1):
                        valid_tokens = sentence_tokens
                    else:
                        # 取邻近的10个单词
                        valid_start_idx = start_token - int(windows_size / 2)
                        valid_end_idx = end_token + int(windows_size / 2)
                        if valid_start_idx < 0:
                            valid_start_idx = 0
                            if valid_end_idx - valid_start_idx > len(sentence_tokens):
                                valid_end_idx = len(sentence_tokens) - 1
                            else:
                                valid_end_idx = valid_end_idx - valid_start_idx
                        if valid_end_idx >= len(sentence_tokens):
                            tmp_num = valid_end_idx - len(sentence_tokens) + 1
                            valid_end_idx = len(sentence_tokens) - 1
                            if valid_start_idx - tmp_num >= 0:
                                valid_start_idx = valid_start_idx - tmp_num
                            else:
                                valid_start_idx = 0
                        valid_tokens = sentence_tokens[valid_start_idx: valid_end_idx]
                    overlap = list(set(valid_tokens) & set(question_tokens))
                    overlap = delete_stopwords(overlap)
                    sentence_overlap.append(len(overlap))
            all_overlaps.append(sentence_overlap)
        best_indices, best_dists = fix_span(context_text, offsets=sentence_offsets, span=qa_info["answer"])
        # 有重叠的和能找到结果数量是一致的
        if len(sum(all_overlaps, [])) != len(best_dists):
            print("error in getting best dist in qas_id:{} overlap num: {}, best dist num:{}".format(
                qa_info['_id'],
                len(sum(all_overlaps, [])),
                len(best_dists)))
        assert len(sum(all_overlaps, [])) == len(best_dists) or len(best_dists) == 1, "error in getting best dist"
        if len(best_dists) > 1:
            min_dist = min(best_dists)
            max_overlap = 0
            new_dists = best_dists
            new_indices = best_indices
            for dist, get_index, overlap_num in zip(best_dists, best_indices, sum(all_overlaps, [])):
                if dist == min_dist and overlap_num > max_overlap:
                    max_overlap = overlap_num
                    new_dists = [dist, ]
                    new_indices = [get_index, ]
            best_dists = new_dists
            best_indices = new_indices
        # 以获取一个结果的方式进行
        for get_index in best_indices:
            start_position = None
            end_position = None
            get_title = None
            for title, idx_map in context_idx2para_idx.items():
                if get_index[0] in idx_map and get_index[1] - 1 in idx_map:
                    start_position = idx_map[get_index[0]]
                    end_position = idx_map[get_index[1] - 1] + 1
                    get_title = title
            if qa_info['_id'] == '5a7a06935542990198eaf050':
                print("error")
            assert start_position is not None and end_position is not None, "error in id: {}".format(qa_info['_id'])
            answers.append([get_title, start_position, end_position])
        qa_info['labels'] = answers
        pre_process_data.append(qa_info)

        for answer in answers:
            for context_info in qa_info["context"]:
                if answer[0] == context_info[0]:
                    if ''.join(context_info[1])[answer[1]: answer[2]] != qa_info["answer"]:
                        print("answer in qas_id: {} is different origin is: {} preprocessed is {}".format(
                            qa_info["_id"], ''.join(context_info[1])[answer[1]: answer[2]], qa_info["answer"]
                        ))
                        different_num += 1
                        break
    print("get {} processed data and has {} different answer and {} has no answer!".format(
        len(pre_process_data),
        different_num,
        all_no_answer_num)
    )
    json.dump(pre_process_data, open(config.preprocessed_file, 'w', encoding='utf-8'))