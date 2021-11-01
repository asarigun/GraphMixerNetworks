import torch.nn as nn
import torch.nn.functional as F
import dgl
from layers_zinc import *

from torchdrug import core, layers
from torchdrug.core import Registry as R
#from scalers import SCALERS
from torch.nn.parameter import Parameter
#from aggregators import AGGREGATORS
#from memory_profiler import profile

class MMAConv(nn.Module, core.Configurable):


    """
    Multi-Masked Aggeragtors for Graph Convolutional Network 
    Parameters:
        input_dim (int): input dimension
        hidden_dims (list of int): hidden dimensions
        edge_input_dim (int, optional): dimension of edge features
        short_cut (bool, optional): use short cut or not
        batch_norm (bool, optional): apply batch normalization or not
        activation (str or function, optional): activation function
        concat_hidden (bool, optional): concat hidden representations from all layers as output
        readout (str, optional): readout function. Available functions are ``sum`` and ``mean``.
    """

    def __init__(self, input_dim, hidden_dims, edge_input_dim=None, short_cut=False, batch_norm=False,
                 activation="relu", concat_hidden=False):

        super(MMAConv, self).__init__()

        #self.weight0 = nn.Parameter(torch.empty(input_dim, hidden_dims))
        #self.bias0 = nn.Parameter(torch.empty(hidden_dims))

        #self.weight1 = nn.Parameter(torch.empty(hidden_dims, 2))
        #self.bias1 = nn.Parameter(torch.empty(2))

        self.weights_mask0 =nn.Parameter(torch.zeros(2*hidden_dims, hidden_dims))
        self.parameters = nn.ParameterList([self.weights_mask0])

        self.input_dim = input_dim
        self.output_dim = hidden_dims[-1] * (len(hidden_dims) if concat_hidden else 1)
        self.dims = [input_dim] + list(hidden_dims)
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden

        self.layers = nn.ModuleList()
        for i in range(len(self.dims)-1):
            self.layers.append(GraphConvolution(self.dims[i], self.dims[i + 1], edge_input_dim, batch_norm))
            self.layers.append(MMA(self.dims[i], self.dims[i + 1], self.weights_mask0, edge_input_dim=None, batch_norm=False))

        self.regressor1 = nn.Linear(hidden_dims, hidden_dims//2)
        self.regressor2 = nn.Linear(hidden_dims//2, 1)

        if readout == "sum":
            self.readout = layers.SumReadout()
        elif readout == "mean":
            self.readout = layers.MeanReadout()
        else:
            raise ValueError("Unknown readout `%s`" % readout)
        

        """
        self.device = device
        self.weight0 = nn.Parameter(torch.cuda.FloatTensor(nfeat, nhid))#.to(self.device)
        self.bias0 = nn.Parameter(torch.cuda.FloatTensor(nhid))#.to(self.device)

        self.weight1 = nn.Parameter(torch.cuda.FloatTensor(nhid))#, nclass))#.to(self.device)
        self.bias1 = nn.Parameter(torch.cuda.FloatTensor(nhid))#.to(self.device)
 
        self.weight_moment_3 = nn.Parameter(torch.cuda.FloatTensor(2*nhid, nhid))#.to(self.device)
        self.weight_sum = nn.Parameter(torch.cuda.FloatTensor(2*nhid, nhid))#.to(self.device)
        self.weight_mean = nn.Parameter(torch.cuda.FloatTensor(2*nhid, nhid))#.to(self.device)
        self.weight_max = nn.Parameter(torch.cuda.FloatTensor(2*nhid, nhid))#.to(self.device)
        self.weight_min = nn.Parameter(torch.cuda.FloatTensor(2*nhid, nhid))#.to(self.device)
        self.weight_softmax = nn.Parameter(torch.cuda.FloatTensor(2*nhid, nhid))#.to(self.device)
        self.weight_softmin = nn.Parameter(torch.cuda.FloatTensor(2*nhid, nhid))#.to(self.device)
        self.weight_std = nn.Parameter(torch.cuda.FloatTensor(2*nhid, nhid))#.to(self.device)
        self.weight_normalized_mean = nn.Parameter(torch.cuda.FloatTensor(2*nhid, nhid))#.to(self.device)

        self.parameters = nn.ParameterList([self.weight0, self.bias0, self.weight1, self.bias1, self.weight_moment_3, self.weight_sum, self.weight_mean, self.weight_max, self.weight_min, self.weight_softmax, self.weight_softmin, self.weight_std, self.weight_normalized_mean])        

        self.g = g
        self.num_atom_type = num_atom_type
        self.num_bond_type = num_bond_type
        self.embed = nn.Embedding(num_atom_type, nfeat)
        
        self.node_emb = Embedding(21, 75)
        self.edge_emb = Embedding(4, 50)

        self.add_all = add_all
        
        self.gc1 = GraphConvolution(nfeat, nhid, self.weight0, self.bias0, device)  
        self.gc2 = MMA(self.add_all, nhid, nhid, self.weight1, self.bias1, self.weight_moment_3, self.weight_sum, self.weight_mean, self.weight_max, self.weight_min, self.weight_softmax, self.weight_softmin, self.weight_std, self.weight_normalized_mean, dropout, aggregator_list, device)  
        self.dropout = dropout

        self.regressor1 = nn.Linear(nhid, nhid//2)
        self.regressor2 = nn.Linear(nhid//2, 1)
        """
    

    def forward(self, graph, input, all_loss=None, metric=None):
        """
        Compute the node representations and the graph representation(s).
        Parameters:
            graph (Graph): :math:`n` graph(s)
            input (Tensor): input node representations
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict
        Returns:
            dict with ``node_feature`` and ``graph_feature`` fields:
                node representations of shape :math:`(|V|, d)`, graph representations of shape :math:`(n, d)`
        """
        hiddens = []
        layer_input = input

        for layer in self.layers:
            hidden = layer(graph, layer_input)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            layer_input = hidden

        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1]
        graph_feature = self.readout(graph, node_feature)
        h = torch.relu(graph_feature)

        h = self.regressor1(h)
        h = torch.relu(h)
        logits = self.regressor2(h)

        return logits

    #def forward(self, graph, input, all_loss=None, metric=None):
        """
        Compute the node representations and the graph representation(s).
        Parameters:
            graph (Graph): :math:`n` graph(s)
            input (Tensor): input node representations
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict
        Returns:
            dict with ``node_feature`` and ``graph_feature`` fields:
                node representations of shape :math:`(|V|, d)`, graph representations of shape :math:`(n, d)`
        """
    #    hiddens = []
    #    layer_input = input

    #    for layer in self.layers:
    #        hidden = layer(graph, layer_input)
    #        if self.short_cut and hidden.shape == layer_input.shape:
    #            hidden = hidden + layer_input
    #        hiddens.append(hidden)
    #        layer_input = hidden

    #    if self.concat_hidden:
    #        node_feature = torch.cat(hiddens, dim=-1)
    #    else:
    #        node_feature = hiddens[-1]
    #    graph_feature = self.readout(graph, node_feature)

    #    return {
    #        "graph_feature": graph_feature,
    #        "node_feature": node_feature
    #    }
 
    
    #def forward(self, add_all, g, nfeat, nhid, num_atom_type, num_bond_type, dropout, aggregator_list, device):
        
    #    h = self.embed(x)
    #    h = F.relu(self.gc1(self.g, h))
    #    h = F.dropout(h, self.dropout, training=self.training)
    #    h = h * snorm_n
    #    h = torch.tanh(h)    
    #    h = self.gc2(self.g, h)
    #    h = F.dropout(h, self.dropout, training=self.training)
    #    h = h * snorm_n
    #    h = torch.tanh(h)

    #    self.g.ndata['h'] = h
    #    h = dgl.mean_nodes(self.g, 'h')
    #    h = torch.relu(h)
        
    #    h = self.regressor1(h)
    #    h = torch.relu(h)
    #    logits = self.regressor2(h)
    #    return logits
