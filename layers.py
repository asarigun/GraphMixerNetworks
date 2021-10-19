import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from mixer import Mixer


class GraphMixer(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphMixer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        #self.mixer = Mixer(input = )
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        #support = self.mixer(input)
        print("self.in_features:",self.in_features)
        print("self.out_features:", self.out_features)
        print("input:", input.shape)
        print("adj:", adj.shape)
        print("self.weight:", self.weight.shape)
        support = torch.mm(input, self.weight)
        print("support:", support.shape)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
