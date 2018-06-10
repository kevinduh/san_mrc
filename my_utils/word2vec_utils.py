'''
GloVe embedding functions
Created June, 2017
Author: xiaodl@microsoft.com
'''

import numpy as np
from .tokenizer import normalize_text

def load_glove_vocab(path, glove_dim=300):
    vocab = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            elems = line.split()
            token = normalize_text(' '.join(elems[0:-glove_dim]))
            vocab.add(token)
    return vocab

def build_embedding(path, vocab, glove_dim):
    vocab_size = len(vocab)
    # zero embedding
    emb = np.zeros((vocab_size, glove_dim))
    emb[0] = 0

    w2id = {w: i for i, w in enumerate(vocab)}
    with open(path, encoding="utf8") as f:
        for line in f:
            elems = line.split()
            token = normalize_text(' '.join(elems[0:-glove_dim]))
            if token in w2id:
                emb[w2id[token]] = [float(v) for v in elems[-glove_dim:]]
    return emb
