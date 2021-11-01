from typing import Optional, List, Dict
from torch_geometric.typing import Adj, OptTensor
from torch.nn.parameter import Parameter
import numpy as np
import math
import torch
from torch import Tensor
from torch_scatter import scatter
import torch.nn as nn
from torch.nn import ModuleList, Sequential, ReLU
from torch_geometric.nn.conv import MessagePassing
#from torch_geometric.nn.dense.linear import Linear
from geometric_linear import Linear
from torch_geometric.utils import degree
import torch.nn.functional as F
from scalers import scalers
#from ..inits import reset
import networkx as nx
from torch_geometric.nn.inits import reset

class GCNConv(MessagePassing):
    r"""GCN Layer "
    <https://arxiv.org/abs/2004.05718>`_ paper
    .. math::
        \mathbf{x}_i^{\prime} = \gamma_{\mathbf{\Theta}} \left(
        \mathbf{x}_i, \underset{j \in \mathcal{N}(i)}{\bigoplus}
        h_{\mathbf{\Theta}} \left( \mathbf{x}_i, \mathbf{x}_j \right)
        \right)
    with
    .. math::
        \bigoplus = \underbrace{\begin{bmatrix}
            1 \\
            S(\mathbf{D}, \alpha=1) \\
            S(\mathbf{D}, \alpha=-1)
        \end{bmatrix} }_{\text{scalers}}
        \otimes \underbrace{\begin{bmatrix}
            \mu \\
            \sigma \\
            \max \\
            \min
        \end{bmatrix}}_{\text{aggregators}},
    where :math:`\gamma_{\mathbf{\Theta}}` and :math:`h_{\mathbf{\Theta}}`
    denote MLPs.
    .. note::
        For an example of using :obj:`PNAConv`, see `examples/pna.py
        <https://github.com/rusty1s/pytorch_geometric/blob/master/
        examples/pna.py>`_.
    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        aggregators (list of str): Set of aggregation function identifiers,
            namely :obj:`"sum"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`,
            :obj:`"var"` and :obj:`"std"`.
        scalers: (list of str): Set of scaling function identifiers, namely
            :obj:`"identity"`, :obj:`"amplification"`,
            :obj:`"attenuation"`, :obj:`"linear"` and
            :obj:`"inverse_linear"`.
        deg (Tensor): Histogram of in-degrees of nodes in the training set,
            used by scalers to normalize.
        edge_dim (int, optional): Edge feature dimensionality (in case
            there are any). (default :obj:`None`)
        towers (int, optional): Number of towers (default: :obj:`1`).
        pre_layers (int, optional): Number of transformation layers before
            aggregation (default: :obj:`1`).
        post_layers (int, optional): Number of transformation layers after
            aggregation (default: :obj:`1`).
        divide_input (bool, optional): Whether the input features should
            be split between towers or not (default: :obj:`False`).
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, dropout, device, in_channels: int, out_channels: int,
                 aggregators: List[str], scalers: List[str], deg: Tensor,
                 edge_dim: Optional[int] = None, towers: int = 1,
                 post_layers: int = 1, divide_input: bool = False, **kwargs):

        kwargs.setdefault('aggr', None)
        super(GCNConv, self).__init__(node_dim=0, **kwargs)

        if divide_input:
            assert in_channels % towers == 0
        assert out_channels % towers == 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggregators = aggregators
        self.scalers = scalers
        self.edge_dim = edge_dim
        self.towers = towers
        self.divide_input = divide_input

        self.F_in = in_channels // towers if divide_input else in_channels
        self.F_out = self.out_channels // towers

        deg = deg.to(torch.float)
        self.avg_deg: Dict[str, float] = {
            'lin': deg.mean().item(),
            'log': (deg + 1).log().mean().item(),
            'exp': deg.exp().mean().item(),
        }

        if self.edge_dim is not None:
            self.edge_encoder = Linear(edge_dim, self.F_in)

        
        self.post_nns = ModuleList()
        for _ in range(towers):

            in_channels = (len(aggregators) * len(scalers) + 1) * self.F_in
            modules = [Linear(in_channels, self.F_out)]
            for _ in range(post_layers - 1):
                modules += [ReLU()]
                modules += [Linear(self.F_out, self.F_out)]
            self.post_nns.append(Sequential(*modules))

        self.lin = Linear(out_channels, out_channels)

        self.reset_parameters()
        self.dropout = dropout
        self.Sig = nn.Sigmoid()
        self.device = device
        

        self.all_aggregators = {'sum': self.learnable_sum, 
                                     'mean': self.learnable_mean, 
                                     'max': self.learnable_max, 
                                     'min': self.learnable_min}


        self.AGGREGATORS = dict()

        for aggr in aggregators:
            self.AGGREGATORS[aggr] = self.all_aggregators[aggr]
        
        self.aggregator_list = [self.AGGREGATORS[aggr] for aggr in self.AGGREGATORS]
        

        self.mask = dict()

        for aggr in self.AGGREGATORS:
            self.mask[aggr] = Parameter(torch.FloatTensor(2*in_channels, out_channels)).to(self.device)

        

    def reset_parameters(self):
        
        if self.edge_dim is not None:
            self.edge_encoder.reset_parameters()
        for nn in self.post_nns:
            reset(nn)
        self.lin.reset_parameters()

    def learnable_sum(self, inputs: Tensor, index: Tensor,
                  dim_size: Optional[int] = None) -> Tensor:

        inputs_new_sum = []

        add_all = index.tolist()

        for i in range(len(add_all)):

            aa = torch.gather(inputs, 0, torch.tensor([[i]*inputs.shape[1]])) #node's features
            aa_tile = torch.tile(aa, [len(add_all[i]), 1]) #ilgili node'un featurlarını neigh
            bb_nei_index2 = add_all[i] #find neighborhood
            bb_nei_index2 = np.array([[i]*inputs.shape[1] for i in bb_nei_index2], dtype="int64") #finding ID of each neighborhood
            bb_nei_index2 = torch.tensor(bb_nei_index2)     
            bb_nei = torch.gather(inputs,0, bb_nei_index2) #finding features of neighbors
            cen_nei = torch.cat([aa_tile, bb_nei],1) #(number of neighs, input[1]*2)
            mask0 = torch.mm(cen_nei, self.mask['sum']) 
            mask0 = self.Sig(mask0)
            mask0 = F.dropout(mask0, self.dropout)
                                      
                                      
            new_cen_nei = aa + torch.sum(mask0 * bb_nei, 0, keepdims=True) 
            inputs_new_sum.append(new_cen_nei)                                      
        
        input_new_sum = torch.stack(inputs_new_sum)                                     
        out = torch.squeeze(input_new_sum)
        out[index] = dim_size
        return out
    
    def learnable_mean(self, inputs: Tensor, index: Tensor, edge_index: Adj,
                  dim_size: Optional[int] = None) -> Tensor:
        
        inputs_new_mean = []
        #adj = edge_index
        #print("adj:", adj)
        #add_all = []
        #for i in range(adj.shape[0]):
        #    add_all.append(adj[i].nonzero()[1])

        add_all = edge_index.tolist()
        #print("add_all:", add_all)
        #print("add_all:", len(add_all))
        
        print("add_all:", add_all)

        for i in range(len(add_all)):

            print("inputs:", inputs.shape)
            print("inputs[1]:", inputs[1].shape)
            aa = torch.gather(inputs[1], 0, torch.tensor([[i]*inputs.shape[2]]).to(self.device))
            print("aa:", aa.shape)
            print("[(add_all[i]), 1]:",len([(add_all[i]), 1]))
            aa_tile = torch.tile(aa, [len(add_all[i]), 1]).to(self.device) #expand central 
            print("aa_tile:", aa_tile.shape)
            bb_nei_index2 = add_all[i]
            bb_nei_index2 = np.array([[i]*inputs.shape[1] for i in bb_nei_index2], dtype="int64")
            bb_nei_index2 = torch.tensor(bb_nei_index2).to(self.device)
            print("bb_nei_index2:", bb_nei_index2.shape)
            bb_nei = torch.gather(inputs[1],0, bb_nei_index2).to(self.device)
            print("bb_nei:", bb_nei.shape)
            cen_nei = torch.cat([aa_tile, bb_nei],1).to(self.device)
            print("cen_nei:", cen_nei.shape)
            print("self.mask['mean']:", self.mask['mean'].shape)
            mask0 = torch.mm(cen_nei, self.mask['mean']).to(self.device)
            mask0 = self.Sig(mask0)
            mask0 = F.dropout(mask0, self.dropout)
                                                     
            new_cen_nei = aa + torch.sum(mask0 * bb_nei, 0, keepdims=True) 
               
            D = len(add_all[i]) 
            new_cen_nei_mean = torch.div(new_cen_nei, D).to(self.device)
            inputs_new_mean.append(new_cen_nei_mean)                                      
                           
        input_new_mean = torch.stack(inputs_new_mean).to(self.device)                                     
        out = torch.squeeze(input_new_mean).to(self.device)
        out[index] = dim_size
        return out
                       
                          
    def learnable_max(self, inputs: Tensor, index: Tensor,
                  dim_size: Optional[int] = None) -> Tensor:   
                       
        inputs_new_max = []

        add_all = index.tolist()
                       
        for i in range(len(add_all)):
               
            #index = torch.tensor([[i]*inputs.shape[1]])
            aa = torch.gather(input, 0, torch.tensor([[i]*inputs.shape[1]])).to(self.device)
            aa_tile = torch.tile(aa, [len(add_all[i]), 1]).to(self.device) 
            bb_nei_index2 = add_all[i]
            bb_nei_index2 = np.array([[i]*inputs.shape[1] for i in bb_nei_index2], dtype="int64")
            bb_nei_index2 = torch.tensor(bb_nei_index2).to(self.device)
            bb_nei = torch.gather(input,0, bb_nei_index2).to(self.device)
            cen_nei = torch.cat([aa_tile, bb_nei],1).to(self.device)
            mask0 = torch.mm(cen_nei, self.mask['max']).to(self.device)
            mask0 = self.Sig(mask0)
            mask0 = F.dropout(mask0, self.dropout)
                                                     
            new_cen_nei_max = torch.max(aa , torch.sum(mask0 * bb_nei, 0, keepdims=True)).to(self.device)
            inputs_new_max.append(new_cen_nei_max)                                      
                           
        input_new_max = torch.stack(inputs_new_max).to(self.device)                                  
        out = torch.squeeze(input_new_max).to(self.device)
        out[index] = dim_size
        return out
                      
  
    def learnable_min(self, inputs: Tensor, index: Tensor,
                  dim_size: Optional[int] = None) -> Tensor:
               
        inputs_new_min = []

        add_all = index.tolist()
               
        for i in range(len(add_all)):
               
            #index = torch.tensor([[i]*inputs.shape[1]])
            aa = torch.gather(inputs, 0, torch.tensor([[i]*inputs.shape[1]])).to(self.device)
            aa_tile = torch.tile(aa, [len(add_all[i]), 1]).to(self.device) #expand central 
            bb_nei_index2 = add_all[i]
            bb_nei_index2 = np.array([[i]*inputs.shape[1] for i in bb_nei_index2], dtype="int64")
            bb_nei_index2 = torch.tensor(bb_nei_index2)
            bb_nei = torch.gather(input,0, bb_nei_index2).to(self.device)
            cen_nei = torch.cat([aa_tile, bb_nei],1).to(self.device)
            mask0 = torch.mm(cen_nei, self.mask['min']).to(self.device)
            mask0 = self.Sig(mask0)
            mask0 = F.dropout(mask0, self.dropout)
                                                     
            new_cen_nei_min = torch.min(aa , torch.sum(mask0 * bb_nei, 0, keepdims=True)).to(self.device)
            inputs_new_min.append(new_cen_nei_min)   

                           
        input_new_min = torch.stack(inputs_new_min).to(self.device)                              
        out = torch.squeeze(input_new_min).to(self.device)
        out[index] = dim_size
        return out

    def aggregate(self, inputs: Tensor, index: Tensor, edge_index: Adj,
                  dim_size: Optional[int] = None) -> Tensor:

        
        outs = []
        for aggregator in self.aggregators:
            if aggregator == 'sum':
                out = self.learnable_sum(inputs, index, dim_size)
                #out = scatter(inputs, index, 0, None, dim_size, reduce='sum')
            elif aggregator == 'mean':
                out = self.learnable_mean(inputs, index, edge_index, dim_size)
                #out = scatter(inputs, index, 0, None, dim_size, reduce='mean')
            elif aggregator == 'min':
                out = self.learnable_min(inputs, index, dim_size)
                #out = scatter(inputs, index, 0, None, dim_size, reduce='min')
            elif aggregator == 'max':
                out = self.learnable_max(inputs, index, dim_size)
                #out = scatter(inputs, index, 0, None, dim_size, reduce='max')
            #elif aggregator == 'var' or aggregator == 'std':
            #    mean = scatter(inputs, index, 0, None, dim_size, reduce='mean')
            #    mean_squares = scatter(inputs * inputs, index, 0, None,
            #                           dim_size, reduce='mean')
            #    out = mean_squares - mean * mean
            #    if aggregator == 'std':
            #        out = torch.sqrt(torch.relu(out) + 1e-5)
            else:
                raise ValueError(f'Unknown aggregator "{aggregator}".')
            outs.append(out)
        out = torch.cat(outs, dim=-1)

        deg = degree(index, dim_size, dtype=inputs.dtype)
        deg = deg.clamp_(1).view(-1, 1, 1)

        outs = []
        for scaler in self.scalers:
            if scaler == 'identity':
                pass
            elif scaler == 'amplification':
                out = out * (torch.log(deg + 1) / self.avg_deg['log'])
            elif scaler == 'attenuation':
                out = out * (self.avg_deg['log'] / torch.log(deg + 1))
            elif scaler == 'linear':
                out = out * (deg / self.avg_deg['lin'])
            elif scaler == 'inverse_linear':
                out = out * (self.avg_deg['lin'] / deg)
            else:
                raise ValueError(f'Unknown scaler "{scaler}".')
            outs.append(out)
        return torch.cat(outs, dim=-1)

    """
    def aggregate(self, inputs: Tensor, index: Tensor,
                  dim_size: Optional[int] = None) -> Tensor:

        
        outs = []
        for aggregator in self.aggregators:
            if aggregator == 'sum':
                out = scatter(inputs, index, 0, None, dim_size, reduce='sum')
            elif aggregator == 'mean':
                out = scatter(inputs, index, 0, None, dim_size, reduce='mean')
            elif aggregator == 'min':
                out = scatter(inputs, index, 0, None, dim_size, reduce='min')
            elif aggregator == 'max':
                out = scatter(inputs, index, 0, None, dim_size, reduce='max')
            elif aggregator == 'var' or aggregator == 'std':
                mean = scatter(inputs, index, 0, None, dim_size, reduce='mean')
                mean_squares = scatter(inputs * inputs, index, 0, None,
                                       dim_size, reduce='mean')
                out = mean_squares - mean * mean
                if aggregator == 'std':
                    out = torch.sqrt(torch.relu(out) + 1e-5)
            else:
                raise ValueError(f'Unknown aggregator "{aggregator}".')
            outs.append(out)
        out = torch.cat(outs, dim=-1)

        deg = degree(index, dim_size, dtype=inputs.dtype)
        deg = deg.clamp_(1).view(-1, 1, 1)

        outs = []
        for scaler in self.scalers:
            if scaler == 'identity':
                pass
            elif scaler == 'amplification':
                out = out * (torch.log(deg + 1) / self.avg_deg['log'])
            elif scaler == 'attenuation':
                out = out * (self.avg_deg['log'] / torch.log(deg + 1))
            elif scaler == 'linear':
                out = out * (deg / self.avg_deg['lin'])
            elif scaler == 'inverse_linear':
                out = out * (self.avg_deg['lin'] / deg)
            else:
                raise ValueError(f'Unknown scaler "{scaler}".')
            outs.append(out)
        return torch.cat(outs, dim=-1)
    """
    
    def forward(self, x: Tensor, edge_index: Adj,
                edge_attr: OptTensor = None) -> Tensor:


        if self.divide_input:
            x = x.view(-1, self.towers, self.F_in)
        else:
            x = x.view(-1, 1, self.F_in).repeat(1, self.towers, 1)

        # propagate_type: (x: Tensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)

        out = torch.cat([x, out], dim=-1)
        outs = [nn(out[:, i]) for i, nn in enumerate(self.post_nns)]
        out = torch.cat(outs, dim=1)

        return self.lin(out)

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, towers={self.towers}, '
                f'edge_dim={self.edge_dim})')