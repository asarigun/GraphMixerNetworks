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

#from ..inits import reset
from torch_geometric.nn.inits import reset



def scalers(self, inputs: Tensor, index: Tensor,
                  dim_size: Optional[int] = None) -> Tensor:
                  
        """
        outs = []
        for aggregator in self.aggregators:
            if aggregator == 'sum':
                out = self.learnable_sum(self, inputs, index)
            elif aggregator == 'mean':
                out = self.learnable_mean(self, inputs, index)
            elif aggregator == 'min':
                out = self.learnable_min(self, inputs, index)
            elif aggregator == 'max':
                out = self.learnable_max(self, inputs, index)
            
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
        """

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