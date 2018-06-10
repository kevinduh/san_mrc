'''
Created October, 2017
Author: xiaodl@microsoft.com
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
from torch.nn.utils import weight_norm
from torch.nn.parameter import Parameter
from .common import activation
from .common import init_wrapper
from .dropout_wrapper import DropoutWrapper

class DotProduct(nn.Module):
    def __init__(self, x1_dim, x2_dim, prefix='sim', opt={}, dropout=None):
        super(DotProduct, self).__init__()
        assert x1_dim == x2_dim
        self.opt = opt
        self.prefix = prefix
        self.scale_on = opt.get('{}_scale'.format(self.prefix), False)
        self.scalor = 1.0 / numpy.power(x2_dim, 0.5)

    def forward(self, x1, x2):
        assert x1.size(2) == x2.size(2)
        scores = x1.bmm(x2.transpose(1, 2))
        if self.scale_on:
            scores *= self.scalor
        return scores


class DotProductProject(nn.Module):
    def __init__(self, x1_dim, x2_dim, prefix='sim', opt={}, dropout=None):
        super(DotProductProject, self).__init__()
        self.prefix = prefix
        self.opt = opt
        self.hidden_size = opt.get('{}_hidden_size'.format(self.prefix), 64)
        self.residual_on = opt.get('{}_residual_on'.format(self.prefix), False)
        self.layer_norm_on = opt.get('{}_norm_on'.format(self.prefix), False)
        self.share = opt.get('{}_share'.format(self.prefix), False)
        self.f = activation(opt.get('{}_activation'.format(self.prefix), 'relu'))
        self.scale_on = opt.get('{}_scale_on'.format(self.prefix), False)
        self.dropout = dropout
        x1_in_dim = x1_dim
        x2_in_dim = x2_dim
        out_dim = self.hidden_size
        self.proj_1 = nn.Linear(x1_in_dim, out_dim, bias=False)
        if self.layer_norm_on:
            self.proj_1 = weight_norm(self.proj_1)
        if self.share and x1_in_dim == x2_in_dim:
            self.proj_2 = self.proj_1
        else:
            self.proj_2 = nn.Linear(x2_in_dim, out_dim)
            if self.layer_norm_on:
                self.proj_2 = weight_norm(self.proj_2)

        if self.scale_on:
            self.scalar = Parameter(torch.ones(1,1,1) / (self.hidden_size ** 0.5), requires_grad=False)
        else:
            self.sclalar = Parameter(torch.ones(1,1, self.hidden_size), requires_grad=True)

    def forward(self, x1, x2):
        assert x1.size(2) == x2.size(2)
        if self.dropout:
            x1 = self.dropout(x1)
            x2 = self.dropout(x2)
        x1_flat = x1.contiguous().view(-1, x1.size(2))
        x2_flat = x2.contiguous().view(-1, x2.size(2))
        x1_o = self.f(self.proj_1(x1_flat)).view(x1.size(0), x1.size(1), -1)
        # x2_o = self.f(self.proj_1(x2_flat)).view(x2.size(0), x2.size(1), -1)
        x2_o = self.f(self.proj_2(x2_flat)).view(x2.size(0), x2.size(1), -1)
        if self.scale_on:
            scalar = self.scalar.expand_as(x2_o)
            x2_o = scalar * x2_o
        scores = x1_o.bmm(x2_o.transpose(1, 2))
        return scores


class Bilinear(nn.Module):
    def __init__(self, x1_dim, x2_dim, prefix='sim', opt={}, dropout=None):
        super(Bilinear, self).__init__()
        self.opt = opt
        self.layer_norm_on = opt.get('{}_norm_on'.format(self.prefix), False)
        self.transform_on = opt.get('{}_proj_on'.format(self.prefix), False)
        # self.init = init_wrapper(opt.get('{}_init'.format(self.prefix), ''))
        self.dropout = dropout
        if self.transform_on:
            self.proj = nn.Linear(x1_dim, x2_dim)
            # self.init(self.proj.weight)
            if self.layer_norm_on: self.proj = weight_norm(self.proj)

    def forward(self, x, y):
        if self.dropout:
            x = self.dropout(x)
            y = self.dropout(y)

        proj = self.proj(y) if self.transform_on else y
        if self.dropout:
            proj = self.dropou(proj)
        scores = x.bmm(proj.unsqueeze(2)).squeeze(2)
        return scores


class BilinearSum(nn.Module):
    def __init__(self, x1_dim, x2_dim, prefix='sim', opt={}, dropout=None):
        super(BilinearSum, self).__init__()
        self.x_linear = nn.Linear(x1_dim, 1, bias=False)
        self.y_linear = nn.Linear(x2_dim, 1, bias=False)
        self.layer_norm_on = opt.get('{}_norm_on'.format(self.prefix), False)
        self.init = init_wrapper(opt.get('{}_init'.format(self.prefix), False))
        if self.layer_norm_on:
            self.x_linear = weight_norm(self.x_linear)
            self.y_linear = weight_norm(self.y_linear)

        self.init(self.x_linear.weight)
        self.init(self.y_linear.weight)
        self.dropout = dropout

    def forward(self, x1, x2):
        if self.dropout:
            x1 = self.dropout(x1)
            x2 = self.dropout(x2)

        x1_logits = self.x_linear(x1.contiguous().view(-1, x1.size(-1))).view(x1.size(0), -1, 1)
        x2_logits = self.y_linear(x2.contiguous().view(-1, x2.size(-1))).view(x2.size(0), 1, -1)

        shape = (x1.size(0), x1.size(1), x2.size())
        scores = x1_logits.expand_as(shape) + x2_logits.expand_as(shape)
        return scores


class Trilinear(nn.Module):
    """Used in BiDAF?"""
    def __init__(self, x1_dim, x2_dim, prefix='sim', opt={}, dropout=None):
        super(Trilinear, self).__init__()
        self.prefix = prefix
        self.x_linear = nn.Linear(x1_dim, 1, bias=False)
        self.x_dot_linear = nn.Linear(x1_dim, 1, bias=False)
        self.y_linear = nn.Linear(x2_dim, 1, bias=False)
        self.layer_norm_on = opt.get('{}_norm_on'.format(self.prefix), False)
        self.init = init_wrapper(opt.get('{}_init'.format(self.prefix), 'xavier_uniform'))
        if self.layer_norm_on:
            self.x_linear = weight_norm(self.x_linear)
            self.x_dot_linear = weight_norm(self.x_dot_linear)
            self.y_linear = weight_norm(self.y_linear)

        self.init(self.x_linear.weight)
        self.init(self.x_dot_linear.weight)
        self.init(self.y_linear.weight)
        self.dropout = dropout

    def forward(self, x1, x2):
        if self.dropout:
            x1 = self.dropout(x1)
            x2 = self.dropout(x2)

        x1_logits = self.x_linear(x1.contiguous().view(-1, x1.size(-1))).view(x1.size(0), -1, 1)
        x2_logits = self.y_linear(x2.contiguous().view(-1, x2.size(-1))).view(x2.size(0), 1, -1)
        x1_dot = self.x_dot_linear(x1.contiguous().view(-1, x1.size(-1))).view(x1.size(0), -1, 1).expand_as(x1)
        x1_dot = x1 * x1_dot

        scores = x1_dot.bmm(x2.transpose(1, 2))
        scores += x1_logits.expand_as(scores) + x2_logits.expand_as(scores)
        return scores


class SimilarityWrapper(nn.Module):
    def __init__(self, x1_dim, x2_dim, prefix='attention', opt={}, dropout=None):
        super(SimilarityWrapper, self).__init__()
        self.score_func_str = opt.get('{}_sim_func'.format(prefix), 'dotproductproject').lower()
        self.score_func = None
        if self.score_func_str == 'dotproduct':
            self.score_func = DotProduct(x1_dim, x2_dim, prefix=prefix, opt=opt, dropout=dropout)
        elif self.score_func_str == 'dotproductproject':
            self.score_func = DotProductProject(x1_dim, x2_dim, prefix=prefix, opt=opt, dropout=dropout)
        elif self.score_func_str == 'bilinear':
            self.score_func = Bilinear(x1_dim, x2_dim, prefix=prefix, opt=opt, dropout=dropout)
        elif self.score_func_str == 'bilinearsum':
            self.score_func = BilinearSum(x1_dim, x2_dim, prefix=prefix, opt=opt, dropout=dropout)
        elif self.score_func_str == 'trilinear':
            self.score_func = Trilinear(x1_dim, x2_dim, prefix=prefix, opt=opt, dropout=dropout)
        else:
            raise NotImplementedError

    def forward(self, x1, x2):
        scores = self.score_func(x1, x2)
        return scores


class AttentionWrapper(nn.Module):
    def __init__(self, x1_dim, x2_dim, prefix='attention', opt={}, dropout=None):
        super(AttentionWrapper, self).__init__()
        self.prefix = prefix
        self.att_dropout = opt.get('{}_att_dropout'.format(self.prefix), 0)
        self.score_func = SimilarityWrapper(x1_dim, x2_dim, prefix=prefix, opt=opt, dropout=dropout)
        self.drop_diagonal = opt.get('{}_drop_diagonal'.format(self.prefix), False)

    def forward(self, x1, x2, x2_mask, x3=None, return_scores=False):
        logits = self.score_func(x1, x2)
        x2_mask = x2_mask.unsqueeze(1).expand_as(logits)
        logits.data.masked_fill_(x2_mask.data, -float('inf'))
        if self.drop_diagonal:
            assert logits.size(1) == logits.size(2)
            diag_mask = torch.diag(logits.data.new(logits.size(1)).zero_() + 1).byte().unsqueeze(0).expand_as(logits)
            logits.data.masked_fill_(diag_mask, -float('inf'))

        prob = F.softmax(logits.view(-1, x2.size(1)), 1)
        prob = prob.view(-1, x1.size(1), x2.size(1))
        if self.att_dropout > 0:
            prob = self.dropout(prob)

        if x3 is None:
            x3 = x2
        att_x1 = prob.bmm(x3)
        if return_scores:
            return att_x1, prob, logits
        else:
            return att_x1


class LinearSelfAttn(nn.Module):
    def __init__(self, input_size, dropout=None):
        super(LinearSelfAttn, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.dropout = dropout

    def forward(self, x, x_mask):
        x = self.dropout(x)
        x_flat = x.contiguous().view(-1, x.size(-1))
        scores = self.linear(x_flat).view(x.size(0), x.size(1))
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(scores, 1)
        return alpha.unsqueeze(1).bmm(x).squeeze(1)


class MLPSelfAttn(nn.Module):
    def __init__(self, input_size, opt={}, prefix='attn_sum', dropout=None):
        super(MLPSelfAttn, self).__init__()
        self.prefix = prefix
        self.FC = nn.Linear(input_size, input_size)
        self.linear = nn.Linear(input_size, 1)
        self.layer_norm_on = opt.get('{}_norm_on'.format(self.prefix), False)
        self.f = activation(opt.get('{}_activation'.format(self.prefix), 'relu'))
        if dropout is None:
            self.dropout = DropoutWrapper(opt.get('{}_dropout_p'.format(self.prefix), 0))
        else:
            self.dropout = dropout
        if self.layer_norm_on:
            self.FC = weight_norm(self.FC)

    def forward(self, x, x_mask):
        x = self.dropout(x)
        x_flat = x.contiguous().view(-1, x.size(-1))
        scores = self.linear(self.f(self.FC(x_flat))).view(x.size(0), x.size(1))
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(scores)
        return alpha.unsqueeze(1).bmm(x).squeeze(1)


class SelfAttnWrapper(nn.Module):
    def __init__(self, input_size, prefix='attn_sum', opt={}, dropout=None):
        super(SelfAttnWrapper, self).__init__()
        attn_type = opt.get('{}_type'.format(prefix), 'linear')
        if attn_type == 'mlp':
            self.att = MLPSelfAttn(input_size, prefix, opt, dropout)
        else:
            self.att = LinearSelfAttn(input_size, dropout)

    def forward(self, x, x_mask):
        return self.att(x, x_mask)


class DeepAttentionWrapper(nn.Module):
    def __init__(self, x1_dim, x2_dim, att_cnt, prefix='deep_att', opt=None, dropout=None):
        super(DeepAttentionWrapper, self).__init__()
        self.opt = {} if opt is None else opt
        self.prefix = prefix
        self.x1_dim = x1_dim
        self.x2_dim = x2_dim
        if dropout is None:
            self.dropout = DropoutWrapper(opt.get('{}_dropout_p'.format(self.prefix), 0))
        else:
            self.dropout = dropout

        self.attn_list = nn.ModuleList()
        for i in range(0, att_cnt):
            attention = AttentionWrapper(self.x1_dim, self.x2_dim, prefix, opt, self.dropout)
            self.attn_list.append(attention)

    def forward(self, x1, x2, x3, x2_mask):
        rvl = []
        for i in range(0, len(x3)):
            hiddens = self.attn_list[i](x1, x2, x2_mask, x3=x3[i])
            rvl.append(hiddens)

        return torch.cat(rvl, 2)


class BilinearFlatSim(nn.Module):
    def __init__(self, x_size, y_size, opt={}, prefix='seqatt', dropout=None):
        super(BilinearFlatSim, self).__init__()
        self.opt = opt
        self.weight_norm_on = opt.get('{}_weight_norm_on'.format(prefix), False)
        self.linear = nn.Linear(y_size, x_size)
        if self.weight_norm_on:
            self.linear = weight_norm(self.linear)
        if dropout is None:
            self.dropout = DropoutWrapper(opt.get('{}_dropout_p'.format(self.prefix), 0))
        else:
            self.dropout = dropout

    def forward(self, x, y, x_mask):
        x = self.dropout(x)
        y = self.dropout(y)

        Wy = self.linear(y)
        xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)
        xWy.data.masked_fill_(x_mask.data, -float('inf'))
        return xWy


class FlatSim(nn.Module):
    def __init__(self, x_size, y_size, opt={}, prefix='seqatt', dropout=None):
        super(FlatSim, self).__init__()
        assert x_size == y_size
        self.opt = opt
        self.weight_norm_on = opt.get('{}_weight_norm_on'.format(prefix), False)
        self.linear = nn.Linear(x_size * 3, 1)
        if self.weight_norm_on:
            self.linear = weight_norm(self.linear)
        if dropout is None:
            self.dropout = DropoutWrapper(opt.get('{}_dropout_p'.format(self.prefix), 0))
        else:
            self.dropout = dropout

    def forward(self, x, y, x_mask):
        x = self.dropout(x)
        y = self.dropout(y)
        y = y.unsqueeze(1).expand_as(x)

        flat_x = torch.cat([x, y, x * y], 2).contiguous().view(x.size(0) * x.size(1), -1)
        flat_scores = self.linear(flat_x)
        scores = flat_scores.contiguous().view(x.size(0), -1)
        scores.data.masked_fill_(x_mask.data, -float('inf'))

        return scores


class FlatSimilarityWrapper(nn.Module):
    def __init__(self, x1_dim, x2_dim, prefix='attention', opt={}, dropout=None):
        super(FlatSimilarityWrapper, self).__init__()
        self.score_func_str = opt.get('{}_att_type'.format(prefix), 'none').lower()
        self.att_dropout = DropoutWrapper(opt.get('{}_att_dropout'.format(prefix), 0))
        self.score_func = None
        if self.score_func_str == 'bilinear':
            self.score_func = BilinearFlatSim(x1_dim, x2_dim, prefix=prefix, opt=opt, dropout=dropout)
        else:
            self.score_func = FlatSim(x1_dim, x2_dim, prefix=prefix, opt=opt, dropout=dropout)

    def forward(self, x1, x2, mask):
        scores = self.score_func(x1, x2, mask)
        return scores