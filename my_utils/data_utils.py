import re
import os
import numpy as np
import logging
import tqdm
import json
from functools import partial
from collections import Counter
from my_utils.tokenizer import Vocabulary, reform_text
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def compute_acc(score_list, gold, threshold=0.5):
    correct = 0
    for key, val in score_list.items():
        lab = 1 if val > threshold else 0
        if lab == gold[key]: correct += 1
    return correct * 100.0 / len(gold)

def gen_name(dir, path, version, suffix='json'):
    fname = '{}_{}.{}'.format(path, version, suffix)
    return os.path.join(dir, fname)

def gen_gold_name(dir, path, version, suffix='json'):
    fname = '{}-{}.{}'.format(path, version, suffix)
    return os.path.join(dir, fname)

def predict_squad(model, data, v2_on=False):
    data.reset()
    span_predictions = {}
    label_predictions = {}
    for batch in data:
        phrase, spans, scores = model.predict(batch)
        uids = batch['uids']
        for uid, pred in zip(uids, phrase):
            span_predictions[uid] = pred
        if v2_on:
            for uid, pred in zip(uids, scores):
                label_predictions[uid] = pred
    return span_predictions, label_predictions

def load_squad_v2_label(path):
    rows = {}
    with open(path, encoding="utf8") as f:
        data = json.load(f)['data']
    for article in tqdm.tqdm(data, total=len(data)):
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                uid, question = qa['id'], qa['question']
                is_impossible = qa.get('is_impossible', False)
                label = 1 if is_impossible else 0
                rows[uid] = label
    return rows

def postag_func(toks, vocab):
    return [vocab[w.tag_] for w in toks if len(w.text) > 0]

def nertag_func(toks, vocab):
    return [vocab['{}_{}'.format(w.ent_type_, w.ent_iob_)] for w in toks if len(w.text) > 0]

def tok_func(toks, vocab, doc_toks=None):
    return [vocab[w.text] for w in toks if len(w.text) > 0]

def raw_txt_func(toks):
    return [w.text for w in toks if len(w.text) > 0]

def match_func(question, context):
    counter = Counter(w.text.lower() for w in context)
    total = sum(counter.values())
    freq = [counter[w.text.lower()] / total for w in context]
    question_word = {w.text for w in question}
    question_lower = {w.text.lower() for w in question}
    question_lemma = {w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower() for w in question}
    match_origin = [1 if w in question_word else 0 for w in context]
    match_lower = [1 if w.text.lower() in question_lower else 0 for w in context]
    match_lemma = [1 if (w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower()) in question_lemma else 0 for w in context]
    features = np.asarray([freq, match_origin, match_lower, match_lemma], dtype=np.float32).T.tolist()
    return features

def build_span(context, answer, context_token, answer_start, answer_end, is_train=True):
    p_str = 0
    p_token = 0
    t_start, t_end, t_span = -1, -1, []
    while p_str < len(context):
        if re.match('\s', context[p_str]):
            p_str += 1
            continue
        token = context_token[p_token]
        token_len = len(token)
        if context[p_str:p_str + token_len] != token:
            return (None, None, [])
        t_span.append((p_str, p_str + token_len))
        if is_train:
            if (p_str <= answer_start and answer_start < p_str + token_len):
                t_start = p_token
            if (p_str < answer_end and answer_end <= p_str + token_len):
                t_end = p_token
        p_str += token_len
        p_token += 1
    if is_train and (t_start == -1 or t_end == -1):
        return (-1, -1, [])
    else:
        return (t_start, t_end, t_span)

def feature_func(sample, query_tokend, doc_tokend, vocab, vocab_tag, vocab_ner, is_train, v2_on=False):
    # features
    fea_dict = {}
    fea_dict['uid'] = sample['uid']
    if v2_on and is_train:
        fea_dict['label'] = sample['label']
    fea_dict['query_tok'] = tok_func(query_tokend, vocab)
    fea_dict['query_pos'] = postag_func(query_tokend, vocab_tag)
    fea_dict['query_ner'] = nertag_func(query_tokend, vocab_ner)
    fea_dict['doc_tok'] = tok_func(doc_tokend, vocab)
    fea_dict['doc_pos'] = postag_func(doc_tokend, vocab_tag)
    fea_dict['doc_ner'] = nertag_func(doc_tokend, vocab_ner)
    fea_dict['doc_fea'] = '{}'.format(match_func(query_tokend, doc_tokend))
    fea_dict['query_fea'] = '{}'.format(match_func(doc_tokend, query_tokend))
    doc_toks = [t.text for t in doc_tokend if len(t.text) > 0]
    query_toks = [t.text for t in query_tokend if len(t.text) > 0]
    answer_start = sample['answer_start']
    answer_end = sample['answer_end']
    answer = sample['answer']
    fea_dict['doc_ctok'] = doc_toks
    fea_dict['query_ctok'] = query_toks

    start, end, span = build_span(sample['context'], answer, doc_toks, answer_start,
                                    answer_end, is_train=is_train)
    if is_train and (start == -1 or end == -1): return None
    if not is_train:
        fea_dict['context'] = sample['context']
        fea_dict['span'] = span
    fea_dict['start'] = start
    fea_dict['end'] = end
    return fea_dict

def build_data(data, vocab, vocab_tag, vocab_ner, fout, is_train, thread=16, NLP=None, v2_on=False):
    passages = [reform_text(sample['context']) for sample in data]
    passage_tokened = [doc for doc in NLP.pipe(passages, batch_size=1000, n_threads=thread)]
    logger.info('Done with document tokenize')

    question_list = [reform_text(sample['question']) for sample in data]
    question_tokened = [question for question in NLP.pipe(question_list, batch_size=1000, n_threads=thread)]
    logger.info('Done with query tokenize')
    dropped_sample = 0
    with open(fout, 'w', encoding='utf-8') as writer:
        for idx, sample in enumerate(data):
            if idx % 5000 == 0: logger.info('parse {}-th sample'.format(idx))
            feat_dict = feature_func(sample, question_tokened[idx], passage_tokened[idx], vocab, vocab_tag, vocab_ner, is_train, v2_on)
            if feat_dict is not None:
                writer.write('{}\n'.format(json.dumps(feat_dict)))
    logger.info('dropped {} in total {}'.format(dropped_sample, len(data)))
