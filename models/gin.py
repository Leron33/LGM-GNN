import sys
import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, GELU
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATConv,GraphConv, GINConv
import numpy
from torch_geometric.nn.inits import reset
from models.mlp import MLP
import time


class GIN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, cat=True, lin=True):
        super(GIN, self).__init__()

        self.in_channels = in_channels
        self.num_layers = num_layers
        self.out_channels = out_channels
        self.cat = cat
        self.lin = lin

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = MLP(in_channels, out_channels, out_channels, 2, dropout=0.0)
            self.convs.append(GINConv(mlp, train_eps=True))
            # self.convs.append(GCNConv(in_channels, out_channels))
            in_channels = out_channels

        if self.cat:
            in_channels = self.in_channels + num_layers * out_channels
        else:
            in_channels = out_channels

        if self.lin:
            self.out_channels = out_channels
            self.final = Lin(in_channels, out_channels)
        else:
            self.out_channels = in_channels

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.lin:
            self.final.reset_parameters()

    def forward(self, x, edge_index, *args):
        """"""
        xs = [x]

        for conv in self.convs:
            xs += [conv(xs[-1], edge_index)]
        
        x = torch.cat(xs, dim=-1) if self.cat else xs[-1]
        x = self.final(x) if self.lin else x
        
        return x

    def __repr__(self):
        return self.__class__.__name__