import sys
import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, GELU
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATConv,GraphConv, GINConv
import numpy
from scipy.optimize import linear_sum_assignment
from torch_geometric.nn.inits import reset
from models.mlp import MLP
from models.gin import GIN
import time


def masked_softmax(src):
    
    srcmax1 = src - torch.max(src,dim=1,keepdim=True)[0]
    src1 = torch.softmax(srcmax1,dim = 1)

    srcmax1 = src - torch.max(src,dim=0,keepdim=True)[0]    
    src1 = torch.softmax(srcmax1,dim = 0)
    out = torch.nan_to_num(src1)
    
    return out

def sinkhorn(src, itera):
    srcmax1 = src - torch.max(src,dim=1,keepdim=True)[0]
    src1 = torch.softmax(srcmax1,dim = 1)
    for i in range(itera):   
        s2 = torch.sum(src1,dim = i%2,keepdim=True)
        src1 = torch.div(src1,s2)
    out = torch.nan_to_num(src1)
    return out


class GNNGM1(torch.nn.Module):
    def __init__(self, num_layers, gnn_layers, hid):
        super(GNNGM1, self).__init__()

        self.num_layers = num_layers
        self.hid = hid
        self.mlp = torch.nn.ModuleList([])
        self.readout = torch.nn.ModuleList([])
        
        in_channel = 1
        out_channel = hid
        self.init=MLP(in_channel,hid,out_channel,2)
        in_channel = out_channel
        for i in range(num_layers):
            self.readout.append(MLP(in_channel+1,hid,1,2))
            self.mlp.append(MLP(in_channel+6,hid,out_channel,2))
            in_channel = out_channel
        
        
    def reset_parameters(self):
        reset(self.init)
        for i in range(self.num_layers):
            reset(self.readout[i])
            reset(self.mlp[i])
        
    def forward(self, G1, G2,adj1,adj2, *args):
        
        n1 = G1.shape[0]
        n2 = G2.shape[0]
        G1 += torch.eye(n1)
        G2 += torch.eye(n2)
        
        x_1 = torch.sum(G1,dim=1).view(n1,1)
        x_2 = torch.sum(G2,dim=1).view(n2,1)
        
        x_1 = self.init(x_1)
        x_2 = self.init(x_2)
        
        P = torch.zeros((n1,n2))
        L = []
        
        for layeri in range(self.num_layers):
            # z_1 = torch.cat(x_1, dim=-1)
            # z_2 = torch.cat(x_2, dim=-1)
            S = masked_softmax(
                self.readout[layeri](
                    torch.cat([P.unsqueeze(-1),-torch.abs(x_1.view(n1,1,-1)-x_2.view(1,n2,-1))/10],dim=2)
                ).squeeze(-1))
            # S = masked_softmax(torch.mm(x_1, torch.transpose(x_2,0,1)))
            
            L.append(S)
            W = torch.log(n1*S*1.5).detach().numpy()
            P = torch.zeros((n1,n2))
            for i in range(n1):
                for j in range(n2):
                    small_W = W[adj1[i],:][:,adj2[j]]
                    row,col = linear_sum_assignment(-small_W)
                    P[i,j] = sum(small_W[row,col])
            
            if layeri < self.num_layers-1:
                r_1 = torch.randn(n1,6)
                r_2 = S.transpose(-1, -2) @ r_1
                x_1 = torch.cat([x_1,r_1],dim=-1)
                x_2 = torch.cat([x_2,r_2],dim=-1)
                # x_1,x_2 = torch.cat((G1.matmul(x_1),S.matmul(x_2)), dim=1),torch.cat(((S.T).matmul(x_1),G2.matmul(x_2)), dim=1)
                # y_1,y_2 = torch.cat((G1.matmul(x_1),S.matmul(x_2)), dim=1),torch.cat((G2.matmul(x_2),(S.T).matmul(x_1)), dim=1)
                # y_1,y_2 = torch.cat((G1.matmul(x_1),x_1), dim=1),torch.cat((G2.matmul(x_2),x_2), dim=1)
                y_1,y_2 = G1.matmul(x_1), G2.matmul(x_2)
                x_1 = self.mlp[layeri](y_1)
                x_2 = self.mlp[layeri](y_2)
            
        
        return P, L
    
    def loss(self, S, y):
        
        nll = 0
        EPS = 1e-12
        k = 1
        for Si in S:
            val = Si[y[0], y[1]]
            nll *= 0.7
            nll += torch.sum(-torch.log(val + EPS))
            
        return nll
    def acc(self, S, y):
       
        Sn = S.detach().numpy()
        row, col = linear_sum_assignment(-Sn)
        pred = torch.tensor(col)
        y[0][y[0]>=len(pred)]=0
        correct = sum(pred[y[0]] == y[1])
        return correct 
    
    def __repr__(self):
        return self.__class__.__name__

