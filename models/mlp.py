import torch
from torch.nn import Linear as Lin
import torch.nn.functional as F


class MLP(torch.nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels, num_layers, dropout=0.0):
        super(MLP, self).__init__()

        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout

        self.lins = torch.nn.ModuleList()
        for _ in range(num_layers-1):
            self.lins.append(Lin(in_channels, hid_channels))
            in_channels = hid_channels
        self.lins.append(Lin(in_channels, out_channels))
        
        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins):
            if i == self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = lin(x)
            if i < self.num_layers - 1:
                x = F.leaky_relu(x)
        return x
