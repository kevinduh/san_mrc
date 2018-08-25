'''
Created October, 2017
Author: xiaodl@microsoft.com
'''
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.utils import weight_norm
from torch.nn import AlphaDropout
from functools import wraps
from src.common import activation
from src.similarity import FlatSimilarityWrapper
from src.dropout_wrapper import DropoutWrapper

SMALL_POS_NUM=1.0e-30

def generate_mask(new_data, dropout_p=0.0, is_training=False):
    if not is_training: dropout_p = 0.0
    new_data = (1-dropout_p) * (new_data.zero_() + 1)
    for i in range(new_data.size(0)):
        one = random.randint(0, new_data.size(1)-1)
        new_data[i][one] = 1
    mask = Variable(1.0/(1 - dropout_p) * torch.bernoulli(new_data), requires_grad=False)
    return mask

class SAN(nn.Module):
    def __init__(self, x_size, h_size, opt={}, prefix='answer', dropout=None):
        super(SAN, self).__init__()
        self.prefix = prefix
        self.attn_b  = FlatSimilarityWrapper(x_size, h_size, prefix, opt, dropout)
        self.attn_e  = FlatSimilarityWrapper(x_size, h_size, prefix, opt, dropout)
        self.rnn_type = '{}{}'.format(opt.get('{}_rnn_type'.format(prefix), 'gru').upper(), 'Cell')
        self.rnn =getattr(nn, self.rnn_type)(x_size, h_size)
        self.num_turn = opt.get('{}_num_turn'.format(prefix), 5)
        self.opt = opt
        self.mem_random_drop = opt.get('{}_mem_drop_p'.format(prefix), 0)
        self.answer_opt = opt.get('{}_opt'.format(prefix), 0)
        self.mem_type = opt.get('{}_mem_type'.format(prefix), 0)
        self.gamma = opt.get('{}_mem_gamma'.format(prefix), 0.5)
        self.alpha = Parameter(torch.zeros(1, 1, 1))

        self.proj = nn.Linear(h_size, x_size) if h_size != x_size else None
        if dropout is None:
            self.dropout = DropoutWrapper(opt.get('{}_dropout_p'.format(self.prefix), 0))
        else:
            self.dropout = dropout

    def forward(self, x, h0, x_mask):
        start_scores_list = []
        end_scores_list = []
        for turn in range(self.num_turn):
            st_scores = self.attn_b(x, h0, x_mask)
            start_scores_list.append(st_scores)
            if self.answer_opt == 3:
                ptr_net_b = torch.bmm(F.softmax(st_scores, 1).unsqueeze(1), x).squeeze(1)
                ptr_net_b = self.dropout(ptr_net_b)
                xb = ptr_net_b if self.proj is None else self.proj(ptr_net_b)
                end_scores = self.attn_e(x, h0 + xb, x_mask)
                ptr_net_e = torch.bmm(F.softmax(end_scores, 1).unsqueeze(1), x).squeeze(1)
                ptr_net_in = (ptr_net_b + ptr_net_e)/2.0
                h0 = self.dropout(h0)
                h0 = self.rnn(ptr_net_in, h0)
            elif self.answer_opt == 2:
                ptr_net_b = torch.bmm(F.softmax(st_scores, 1).unsqueeze(1), x).squeeze(1)
                ptr_net_b = self.dropout(ptr_net_b)
                xb = ptr_net_b if self.proj is None else self.proj(ptr_net_b)
                end_scores = self.attn_e(x, xb, x_mask)
                ptr_net_e = torch.bmm(F.softmax(end_scores, 1).unsqueeze(1), x).squeeze(1)
                ptr_net_in = ptr_net_e
                h0 = self.dropout(h0)
                h0 = self.rnn(ptr_net_in, h0)
            elif self.answer_opt == 1:
                ptr_net_b = torch.bmm(F.softmax(st_scores, 1).unsqueeze(1), x).squeeze(1)
                ptr_net_b = self.dropout(ptr_net_b)
                h0 = self.dropout(h0)
                ptr_net_in = ptr_net_b
                h1 = self.rnn(ptr_net_in, h0)
                end_scores = self.attn_e(x, h1, x_mask)
                h0 = h1
            else:
                end_scores = self.attn_e(x, h0, x_mask)
                ptr_net_e = torch.bmm(F.softmax(end_scores, 1).unsqueeze(1), x).squeeze(1)
                ptr_net_in = ptr_net_e
                h0 = self.dropout(h0)
                h0 = self.rnn(ptr_net_in, h0)
            end_scores_list.append(end_scores)

        if self.mem_type == 1:
            mask = generate_mask(self.alpha.data.new(x.size(0), self.num_turn), self.mem_random_drop, self.training)
            mask = [m.contiguous() for m in torch.unbind(mask, 1)]
            start_scores_list = [mask[idx].view(x.size(0), 1).expand_as(inp) * F.softmax(inp, 1) for idx, inp in enumerate(start_scores_list)]
            end_scores_list = [mask[idx].view(x.size(0), 1).expand_as(inp) * F.softmax(inp, 1) for idx, inp in enumerate(end_scores_list)]
            start_scores = torch.stack(start_scores_list, 2)
            end_scores = torch.stack(end_scores_list, 2)
            start_scores = torch.mean(start_scores, 2)
            end_scores = torch.mean(end_scores, 2)
            start_scores.data.masked_fill_(x_mask.data, SMALL_POS_NUM)
            end_scores.data.masked_fill_(x_mask.data, SMALL_POS_NUM)
            start_scores = torch.log(start_scores)
            end_scores = torch.log(end_scores)
        else:
            start_scores = start_scores_list[-1]
            end_scores = end_scores_list[-1]

        return start_scores, end_scores
