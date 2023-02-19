import spacy
import bisect
import re

nlp = spacy.blank("en")


def find_nearest(a, target, test_func=lambda x: True):
    idx = bisect.bisect_left(a, target)
    if (0 <= idx < len(a)) and a[idx] == target:
        return target, 0
    elif idx == 0:
        return a[0], abs(a[0] - target)
    elif idx == len(a):
        return a[-1], abs(a[-1] - target)
    else:
        d1 = abs(a[idx] - target) if test_func(a[idx]) else 1e200
        d2 = abs(a[idx-1] - target) if test_func(a[idx-1]) else 1e200
        if d1 > d2:
            return a[idx-1], d2
        else:
            return a[idx], d1


def fix_span(para, offsets, span):
    span = span.strip()
    parastr = "".join(para)
    assert span in parastr, '{}\t{}'.format(span, parastr)
    begins, ends = map(list, zip(*[y for x in offsets for y in x]))

    best_dist = 1e200
    best_dists = []
    best_indices = []

    if span == parastr:
        return parastr, (0, len(parastr)), 0

    for m in re.finditer(re.escape(span), parastr):
        begin_offset, end_offset = m.span()

        fixed_begin, d1 = find_nearest(begins, begin_offset, lambda x: x < end_offset)
        fixed_end, d2 = find_nearest(ends, end_offset, lambda x: x > begin_offset)

        best_dists.append(d1 + d2)
        best_indices.append((fixed_begin, fixed_end))

    # assert best_indices is not None
    return best_indices, best_dists


def word_tokenize(sent):
    """ 使用spacy进行分词"""
    doc = nlp(sent)
    return [token.text for token in doc]


def delete_stopwords(words):
    """ 删除停顿词 """
    punctuation_mark = {',', '.', '?', ';', ':', '(', ')', '!', '\'', '"', ''}
    return [word for word in words if not nlp.vocab[word].is_stop and word not in punctuation_mark]


def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        pre = current
        current = text.find(token, current)
        if current < 0:
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


def prepro_sent(sent):
    return sent
    # return sent.replace("''", '" ').replace("``", '" ')