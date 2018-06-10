'''
Created October, 2017
Author: xiaodl@microsoft.com
'''
import re
import warnings
import spacy
import tqdm
import logging
import unicodedata
from collections import Counter


logger = logging.getLogger(__name__)

PAD = '<PAD>'
UNK = '<UNK>'
STA= '<BOS>'
END = '<EOS>'

PAD_ID = 0
UNK_ID = 1
STA_ID = 2
END_ID = 3

def normalize_text(text):
    return unicodedata.normalize('NFD', text)

def token_extend(reg_rules):
    return ' ' + reg_rules.group(0) + ' '

def reform_text(text):
    text = re.sub(u'-|¢|¥|€|£|\u2010|\u2011|\u2012|\u2013|\u2014|\u2015|%|\[|\]|:|\(|\)|/', token_extend, text)
    text = text.strip(' \n')
    text = re.sub('\s+', ' ', text)
    return text

class Vocabulary(object):
    INIT_LEN = 4
    def __init__(self, neat=False):
        self.neat = neat
        if not neat:
            self.tok2ind = {PAD: PAD_ID, UNK: UNK_ID, STA: STA_ID, END: END_ID}
            self.ind2tok = {PAD_ID: PAD, UNK_ID: UNK, STA_ID: STA, END_ID:END}
        else:
            self.tok2ind = {}
            self.ind2tok = {}

    def __len__(self):
        return len(self.tok2ind)

    def __iter__(self):
        return iter(self.tok2ind)

    def __contains__(self, key):
        if type(key) == int:
            return key in self.ind2tok
        elif type(key) == str:
            return key in self.tok2ind

    def __getitem__(self, key):
        if type(key) == int:
            return self.ind2tok.get(key, -1) if self.neat else self.ind2tok.get(key, UNK)
        if type(key) == str:
            return self.tok2ind.get(key, None) if self.neat else self.tok2ind.get(key,self.tok2ind.get(UNK))

    def __setitem__(self, key, item):
        if type(key) == int and type(item) == str:
            self.ind2tok[key] = item
        elif type(key) == str and type(item) == int:
            self.tok2ind[key] = item
        else:
            raise RuntimeError('Invalid (key, item) types.')

    def add(self, token):
        if token not in self.tok2ind:
            index = len(self.tok2ind)
            self.tok2ind[token] = index
            self.ind2tok[index] = token

    def get_vocab_list(self, with_order=True):
        if with_order:
            words = [self[k] for k in range(0, len(self))]
        else:
            words = [k for k in self.tok2ind.keys()
                      if k not in {PAD, UNK, STA, END}]
        return words

    def toidx(self, tokens):
        return [self[tok] for tok in tokens]

    def copy(self):
        new_vocab = Vocabulary(self.neat)
        for w in self:
            new_vocab.add(w)
        return new_vocab

    def build(words, neat=False):
        vocab = Vocabulary(neat)
        for w in words: vocab.add(w)
        return vocab
