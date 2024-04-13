import sys
import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, GELU, Sigmoid
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATConv,GraphConv, GINConv
import numpy
from scipy.optimize import linear_sum_assignment
from torch_geometric.nn.inits import reset
from models.mlp import MLP
from models.gin import GIN

from multiprocessing import Pool
from itertools import repeat,product
import time

def masked_softmax(src):
    
    srcmax1 = src - torch.max(src,dim=1,keepdim=True)[0]
    src1 = torch.softmax(srcmax1,dim = 1)
    src1 = torch.nan_to_num(src1)

    srcmax2 = src - torch.max(src,dim=0,keepdim=True)[0]
    src2 = torch.softmax(srcmax2,dim = 0)
    src2 = torch.nan_to_num(src2)
    
    return (src1+src2)/2

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
        out_channel = 2
        for i in range(num_layers):
            self.readout.append(MLP(in_channel,hid,1,2))
            self.mlp.append(MLP(in_channel,hid,hid-1,2))
            in_channel = 1
        self.final = MLP(1,hid,1,2)
        
    def reset_parameters(self):
        reset(self.final)
        for i in range(self.num_layers):
            reset(self.readout[i])
            reset(self.mlp[i])

    def loc_match(self, adj):
        small_W = self.W[adj[0],:][:,adj[1]]
        row,col = linear_sum_assignment(-small_W)
        weight = small_W[row,col]
        return sum(weight)
    
    def forward(self, G1, G2, adj1, adj2, multi_p = 1, *args):
        
        n1 = G1.shape[0]
        n2 = G2.shape[0]
        
        d1 = torch.sum(G1,dim=1)
        d2 = torch.sum(G2,dim=1)
        d_mean = torch.mean(d1)
        ttheta = d_mean
        
        D_diff = -torch.abs(d1.view(n1,1)-d2.view(1,n2)).unsqueeze(-1)
        S = D_diff
        L = []
        
        for layeri in range(self.num_layers):
            
            Wn = sinkhorn(self.readout[layeri](S/ttheta).squeeze(-1),5)
            W = torch.log(Wn*n1*1.5)
            #W[W<-2] = -2
            W[W==float('-Inf')] = -1e12
            W[W==float('Inf')] = 1e12
            
            self.W = W.detach().numpy()

            arg_adj = product(adj1,adj2)
            if multi_p == 1:
                result = list(map(self.loc_match, arg_adj))
            else:
                with Pool(multi_p) as pool:
                    result = pool.map(self.loc_match, arg_adj)
            X_1 = torch.Tensor(result).reshape(n1,n2)
            
            #H = torch.einsum("abh,bc->ach",torch.einsum("ij,jkh->ikh",G1,S),G2)
            #X = self.mlp[layeri](H)
            
            S = torch.zeros((n1,n2,1)) 
            S[:,:,0] = X_1
            #S[:,:,1:] = X
            L.append(Wn)
            
        output = masked_softmax(self.final(S).squeeze(-1))
        L.append(output)
        return output, L#[output]
    
    def loss(self, S, y):
        
        nll = 0
        EPS = 1e-12
        k = 1
        n1 = S[0].shape[0]
        for Si in S:
            val = Si[y[0], y[1]]
            nll *= 0.5
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