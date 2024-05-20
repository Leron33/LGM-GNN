import sys
import os
import numpy as np
from scipy import sparse as sp
from scipy.optimize import linear_sum_assignment
import random

import argparse
import torch
import torch_geometric.transforms as T
from multiprocessing import Pool
from itertools import product
import time

# from GMAlgorithms import SGM as FAQ, PATH, Grampa, NTMA, DP,DP1
torch.set_printoptions(precision=4)

parser = argparse.ArgumentParser()
parser.add_argument('--hid', type=int, default=8)
parser.add_argument('--num_layers', type=int, default=6)
parser.add_argument('--gnn_layers', type=int, default=2)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--gamma', type=float, default=0.8)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--runs', type=int, default=20)

args, unknown = parser.parse_known_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class NTMA():
    def __init__(self,num_layer):
        self.layer = num_layer
        return 
    def loc_match(self, adj):
        small_W = self.W[adj[0],:][:,adj[1]]
        row,col = linear_sum_assignment(-small_W)
        weight = small_W[row,col]
        return sum(weight)
    
    def match(self,G1,G2,adj1,adj2,multi_p=1):
    
        n1 = G1.shape[0]
        n2 = G2.shape[0]
    
        d1 = torch.sum(G1,dim=1).view(n1,1)
        d2 = torch.sum(G2,dim=1).view(1,n2)
        D_diff = d1+d2-torch.abs(d1-d2)
        self.W = D_diff
        output = []
        for _ in range(self.layer):
            arg_adj = product(adj1,adj2)
            if multi_p == 1:
                result = list(map(self.loc_match, arg_adj))
            else:
                with Pool(multi_p) as pool:
                    result = pool.map(self.loc_match, arg_adj)
            self.W = torch.Tensor(result).reshape(n1,n2)
            output.append(self.W)
        
        return output
    
    def acc(self, S, y):
        
        row, col = linear_sum_assignment(-S.detach().numpy())
        pred = torch.tensor(col)
        correct = sum((pred[y[0]]==y[1]).float())/len(pred)
        return correct 


ntma_v = NTMA(4)
# model = GNNGM(args.num_layers, args.gnn_layers,args.hid).to(device)

def sinkhorn(src, itera):
    srcmax1 = src - torch.max(src,dim=1,keepdim=True)[0]
    src1 = torch.softmax(srcmax1,dim = 1)
    for i in range(itera):   
        s2 = torch.sum(src1,dim = i%2,keepdim=True)
        src1 = torch.div(src1,s2)
    out = torch.nan_to_num(src1)
    return out

def Correlated_ER_Graph(n, p, s):
    
    a = (torch.rand(n,n)<p).float()
    a = torch.triu(a,diagonal=1)
    G0 = a+a.T
    
    sample = (torch.rand(n,n)<s).float()
    sample = torch.triu(sample,1)
    sample = sample + sample.T
    G1 = G0*sample
    
    sample = (torch.rand(n,n)<s).float()
    sample = torch.triu(sample,1)
    sample = sample + sample.T
    G2 = G0*sample
    
    #truth = torch.randperm(n)
    truth = n-1-torch.arange(n)
    # truth = torch.arange(n)
        
    G1 = G1[truth,:][:,truth]
    
    one_to_n = torch.arange(n)
    y = torch.stack((one_to_n, truth))
    
    adj1 = []
    adj2 = []
        
    for i in range(n):
        adj1.append(torch.nonzero(G1[i,:], as_tuple=True)[0])
    for i in range(n):
        adj2.append(torch.nonzero(G2[i,:], as_tuple=True)[0])
        
    return (G1, G2, adj1, adj2, y)

def run(n,p,S,Itera):
    
    gnn = torch.zeros(Itera,len(S))
    ntma = torch.zeros(Itera,len(S))
    ntma1 = torch.zeros(Itera,len(S))
    ntma2 = torch.zeros(Itera,len(S))
    ntma3 = torch.zeros(Itera,len(S))
    ntma4 = torch.zeros(Itera,len(S))
    dp = torch.zeros(Itera,len(S))

    for si, s in enumerate(S):
        print(s)
        for itera in range(Itera):
            
            G1, G2, adj1, adj2, y = Correlated_ER_Graph(n,p,s)
            
            
            W = ntma_v.match(G1,G2,adj1,adj2)
            
            row, col = linear_sum_assignment(-W[0].detach().numpy())
            result = torch.tensor(col)
            ntma1[itera,si] = sum((result[y[0]]==y[1]).float())/n 
            
            row, col = linear_sum_assignment(-W[1].detach().numpy())
            result = torch.tensor(col)
            ntma2[itera,si] = sum((result[y[0]]==y[1]).float())/n
            
            row, col = linear_sum_assignment(-W[2].detach().numpy())
            result = torch.tensor(col)
            ntma3[itera,si] = sum((result[y[0]]==y[1]).float())/n
            
            row, col = linear_sum_assignment(-W[3].detach().numpy())
            result = torch.tensor(col)
            ntma4[itera,si] = sum((result[y[0]]==y[1]).float())/n
           
            
            # result = Grampa(G1,G2,1)
            # grampa[itera,si] = sum((result[y[0]]==y[1]).float())/n 
            
            # result = DP1(G1,G2,adj1,adj2)
            # dp[itera,si] = sum((result[y[0]]==y[1]).float())/n
            
            

    gnnstd, gnn = torch.std_mean(gnn,dim=0,unbiased=False)
    ntmastd,ntma = torch.std_mean(ntma,dim=0,unbiased=False)
    dpstd,dp = torch.std_mean(dp,dim=0,unbiased=False)
    
    _,ntma1 = torch.std_mean(ntma1,dim=0,unbiased=False)
    _,ntma2 = torch.std_mean(ntma2,dim=0,unbiased=False)
    _,ntma3 = torch.std_mean(ntma3,dim=0,unbiased=False)
    _,ntma4 = torch.std_mean(ntma4,dim=0,unbiased=False)
    
    S = ', '.join(str(round(i, 4)) for i in S)
    gnn = ', '.join(str(round(i, 4)) for i in (gnn).tolist())
    ntma = ', '.join(str(round(i, 4)) for i in (ntma).tolist())
    dp = ', '.join(str(round(i, 4)) for i in (dp).tolist())
    ntma1 = ', '.join(str(round(i, 4)) for i in (ntma1).tolist())
    ntma2 = ', '.join(str(round(i, 4)) for i in (ntma2).tolist())
    ntma3 = ', '.join(str(round(i, 4)) for i in (ntma3).tolist())
    ntma4 = ', '.join(str(round(i, 4)) for i in (ntma4).tolist())
    # with open('TestER2.txt', 'a') as f:
    #     f.write('s = ['+S+']\n')
    #     f.write('GNN = ['+gnn+']\n')
    #     f.write('grampa = ['+grampa+']\n')
    #     f.write('sgm  = ['+sgm+']\n' )
    #     f.write('faqd = ['+ntma+']\n')
    #     f.write('dp = ['+dp+']\n')
    #     f.write('-----------------------------------------\n')

    torch.set_printoptions(precision=4)
    print(f'Parameters: n={n}, p={p}')
    print('Accuracy')
    print('s = '.ljust(10), S)
    print('ntma1 = '.ljust(10), ntma1)
    print('ntma2 = '.ljust(10), ntma2)
    print('ntma3 = '.ljust(10), ntma3)
    print('ntma4 = '.ljust(10), ntma4)

# path = "./model/GNN-5-prod-8.pth"
# model.load_state_dict(torch.load(path))
# open('TestER2.txt', 'w')

n = 1000
p = 0.1
S = [0.99,0.97, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]
Itera = 10
start_time = time.time()
run(n, p, S, Itera)
print("--- %s seconds ---" % (time.time() - start_time))

print('-----------------------------------------------')

n = 1000
p = 0.01
S = [0.99,0.97, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]
Itera = 10
start_time = time.time()
run(n, p, S, Itera)
print("--- %s seconds ---" % (time.time() - start_time))
