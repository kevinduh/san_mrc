'''
Created October, 2017
Author: xiaodl@microsoft.com
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class PositionwiseNN(nn.Module):
    def __init__(self, idim, hdim, dropout=None):
        super(PositionwiseNN, self).__init__()
        self.w_0 = nn.Conv1d(idim, hdim, 1)
        self.w_1 = nn.Conv1d(hdim, hdim, 1)
        self.dropout = dropout

    def forward(self, x):
        output = F.relu(self.w_0(x.transpose(1, 2)))
        output = self.dropout(output)
        output = self.w_1(output)
        output = self.dropout(output).transpose(2, 1)
        return output


class LayerNorm(nn.Module):
    #ref: https://github.com/pytorch/pytorch/issues/1959
    #   :https://arxiv.org/pdf/1607.06450.pdf
    def __init__(self, hidden_size, eps=1e-4):
        super(LayerNorm, self).__init__()
        self.alpha = Parameter(torch.ones(1,1,hidden_size)) # gain g
        self.beta = Parameter(torch.zeros(1,1,hidden_size)) # bias b
        self.eps = eps

    def forward(self, x):
        mu = torch.mean(x, 2, keepdim=True).expand_as(x)
        sigma = torch.std(x, 2, keepdim=True).expand_as(x)
        return (x - mu) / (sigma + self.eps) * self.alpha.expand_as(x) + self.beta.expand_as(x)


class Highway(nn.Module):
    def __init__(self, size, num_layers, f=F.relu):
        super(Highway, self).__init__()
        self.num_layers = num_layers
        self.nonlinear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.f = f

    def forward(self, x):
        for layer in range(self.num_layers):
            gate = F.sigmoid(self.gate[layer](x))
            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)
            x = gate * nonlinear + (1 - gate) * linear
        return x