import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module

import numpy as np
import math

#from scalers import SCALERS
#from aggregators_new import AGGREGATORS
#from memory_profiler import profile

from torch.utils import checkpoint
from torch_scatter import scatter_mean, scatter_add, scatter_max

from torchdrug import data, layers, utils
from torchdrug.layers import functional

##### Taken and Modified from https://github.com/DeepGraphLearning/torchdrug #######

class GraphConvolution(Module):

    
    def __init__(self, in_features, out_features, edge_input_dim=None, batch_norm=False): 
        super(GraphConvolution, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.edge_input_dim = edge_input_dim

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(out_features)
        else:
            self.batch_norm = None

        self.activation = torch.relu()

        self.linear = nn.Linear(in_features, out_features)
        if edge_input_dim:
            self.edge_linear = nn.Linear(edge_input_dim, in_features)
        else:
            self.edge_linear = None

        #self.weight = weight
        
        #self.bias = bias
        

    def message(self, graph, input):
        # add self loop
        node_in = torch.cat([graph.edge_list[:, 0], torch.arange(graph.num_node, device=graph.device)])
        degree_in = graph.degree_in.unsqueeze(-1) + 1
        message = input[node_in]
        if self.edge_linear:
            edge_input = self.edge_linear(graph.edge_feature.float())
            edge_input = torch.cat([edge_input, torch.zeros(graph.num_node, self.input_dim, device=graph.device)])
            message += edge_input
        message /= degree_in[node_in].sqrt()
        return message
     
    def aggregate(self, graph, message):
        # add self loop
        node_out = torch.cat([graph.edge_list[:, 1], torch.arange(graph.num_node, device=graph.device)])
        edge_weight = torch.cat([graph.edge_weight, torch.ones(graph.num_node, device=graph.device)])
        edge_weight = edge_weight.unsqueeze(-1)
        degree_out = graph.degree_out.unsqueeze(-1) + 1
        update = scatter_add(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)
        update = update / degree_out.sqrt()
        return update

    def message_and_aggregate(self, graph, input):
        node_in, node_out = graph.edge_list.t()[:2]
        node_in = torch.cat([node_in, torch.arange(graph.num_node, device=graph.device)])
        node_out = torch.cat([node_out, torch.arange(graph.num_node, device=graph.device)])
        edge_weight = torch.cat([graph.edge_weight, torch.ones(graph.num_node, device=graph.device)])
        degree_in = graph.degree_in + 1
        degree_out = graph.degree_out + 1
        edge_weight = edge_weight / (degree_in[node_in] * degree_out[node_out]).sqrt()
        adjacency = utils.sparse_coo_tensor(torch.stack([node_in, node_out]), edge_weight,
                                            (graph.num_node, graph.num_node))
        update = torch.sparse.mm(adjacency.t(), input)
        if self.edge_linear:
            edge_input = graph.edge_feature.float()
            if self.edge_linear.in_features > self.edge_linear.out_features:
                edge_input = self.edge_linear(edge_input)
            edge_weight = edge_weight.unsqueeze(-1)
            edge_update = scatter_add(edge_input * edge_weight, graph.edge_list[:, 1], dim=0,
                                      dim_size=graph.num_node)
            if self.edge_linear.in_features <= self.edge_linear.out_features:
                edge_update = self.edge_linear(edge_update)
            update += edge_update

        return update

    def combine(self, input, update):
        output = self.linear(update)
        if self.batch_norm:
            output = self.batch_norm(output)
        if self.activation:
            output = self.activation(output)
        return output

    
class MMA(Module):

    def __init__(self, in_features, out_features, weights_mask0, edge_input_dim=None, batch_norm=False): 
        super(MMA, self).__init__()


        self.in_features = in_features
        self.out_features = out_features
        self.edge_input_dim = edge_input_dim

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(out_features)
        else:
            self.batch_norm = None

        self.activation = torch.relu()

        self.linear = nn.Linear(in_features, out_features)
        if edge_input_dim:
            self.edge_linear = nn.Linear(edge_input_dim, in_features)
        else:
            self.edge_linear = None

        self.Sig = nn.Sigmoid()
        self.drop_rate = 0.5
        self.weights_mask0 = weights_mask0
        

    def message(self, graph, input):
        # add self loop
        node_in = torch.cat([graph.edge_list[:, 0], torch.arange(graph.num_node, device=graph.device)])
        degree_in = graph.degree_in.unsqueeze(-1) + 1
        message = input[node_in]
        if self.edge_linear:
            edge_input = self.edge_linear(graph.edge_feature.float())
            edge_input = torch.cat([edge_input, torch.zeros(graph.num_node, self.input_dim, device=graph.device)])
            message += edge_input
        message /= degree_in[node_in].sqrt()
        return message
     
    def aggregate(self, graph, message):
        # add self loop
        node_out = torch.cat([graph.edge_list[:, 1], torch.arange(graph.num_node, device=graph.device)])
        edge_weight = torch.cat([graph.edge_weight, torch.ones(graph.num_node, device=graph.device)])
        edge_weight = edge_weight.unsqueeze(-1)
        degree_out = graph.degree_out.unsqueeze(-1) + 1
        update = scatter_add(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)
        update = update / degree_out.sqrt()
        return update

    def message_and_learnable_sum_aggregate(self, graph, input):
        node_in, node_out = graph.edge_list.t()[:2]
        node_in = torch.cat([node_in, torch.arange(graph.num_node, device=graph.device)])
        node_out = torch.cat([node_out, torch.arange(graph.num_node, device=graph.device)])
        edge_weight = torch.cat([graph.edge_weight, torch.ones(graph.num_node, device=graph.device)])
        degree_in = graph.degree_in + 1
        degree_out = graph.degree_out + 1
        edge_weight = edge_weight / (degree_in[node_in] * degree_out[node_out]).sqrt()
        adjacency = utils.sparse_coo_tensor(torch.stack([node_in, node_out]), edge_weight,
                                            (graph.num_node, graph.num_node))

        add_all = []
        for i in range(adjacency.shape[0]):
            add_all.append(adjacency[i].nonzero()[1])
        
        input_new = []
        for i in range(len(add_all)):
            index = torch.tensor([[i]*input.shape[1]])
            aa = torch.gather(input, 0, torch.tensor([[i]*input.shape[1]]))
            aa_tile = torch.tile(aa, [len(self.add_all[i]), 1])
            bb_nei_index2 = add_all[i]
            bb_nei_index2 = np.array([[i]*input.shape[1] for i in bb_nei_index2], dtype="int64")
            bb_nei_index2 = torch.tensor(bb_nei_index2)
            bb_nei = torch.gather(input,0, bb_nei_index2)
            cen_nei = torch.cat([aa_tile, bb_nei],1)
            mask0 = torch.mm(cen_nei, self.weights_mask0)
            mask0 = self.Sig(mask0)
            mask0 = F.dropout(mask0, self.drop_rate)
            new_cen_nei = aa + torch.sum(mask0 * bb_nei, 0, keepdims=True)
            input_new.append(new_cen_nei)

        input_new = torch.stack(input_new)
        input_new = torch.squeeze(input_new)


        update = torch.sparse.mm(adjacency.t(), input_new)
        if self.edge_linear:
            edge_input = graph.edge_feature.float()
            if self.edge_linear.in_features > self.edge_linear.out_features:
                edge_input = self.edge_linear(edge_input)
            edge_weight = edge_weight.unsqueeze(-1)
            edge_update = scatter_add(edge_input * edge_weight, graph.edge_list[:, 1], dim=0,
                                      dim_size=graph.num_node)
            if self.edge_linear.in_features <= self.edge_linear.out_features:
                edge_update = self.edge_linear(edge_update)
            update += edge_update

        return update

    def combine(self, input, update):
        output = self.linear(update)
        if self.batch_norm:
            output = self.batch_norm(output)
        if self.activation:
            output = self.activation(output)
        return output
