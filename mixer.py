import torch
import torch.nn as nn
#import torch.nn.functional as F
#from einops.layers.torch import Rearrange

class MlpBlock(nn.Module):
    def __init__(self, input, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(input),
            nn.Dropout(p)
        )

    def forward(self, x):
        return self.net(x)

class Mixer(nn.Module):

    def __init__(self, input):
        super().__init__()

        self.norm = nn.LayerNorm()
        self.transpose = torch.transpose()
        self.mlp1 = MlpBlock(input, p=0.1)
        self.mlp2 = MlpBlock(input, p=0.1)
        
    def forward(self, x):
        x_old =x
        x = self.norm(x)
        x = self.transpose(x)
        x = self.mlp1(x)
        x = self.transpose(x)
        x = x + x_old
        x_new = x
        x = self.norm(x)
        x = self.mlp2(x)
        x = x + x_new
        return x