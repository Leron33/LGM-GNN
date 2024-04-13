import sys
import copy
import os
import os.path as osp
import numpy as np
from scipy import sparse as sp
from scipy.optimize import linear_sum_assignment
import random
import matplotlib.pyplot as plt

import argparse
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance
torch.set_printoptions(precision=4)
import math
import time

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

def sinkhorn(src, itera):
    srcmax1 = src - torch.max(src,dim=1,keepdim=True)[0]
    src1 = torch.softmax(srcmax1,dim = 1)
    for i in range(itera):   
        s2 = torch.sum(src1,dim = i%2,keepdim=True)
        src1 = torch.div(src1,s2)
    out = torch.nan_to_num(src1)
    return out

def plotS(S):
    fig = plt.figure(figsize=(6, 6))
    axi = plt.subplot(1,1,1)
    
    S_min = torch.min(S)
    S_max = torch.max(S)
    S = (S - S_min) / (S_max - S_min)
    
    S = torch.rot90(S,3).detach().numpy()
    print(S)
    sns.heatmap(data=list(S),cbar=False,
                cmap=plt.get_cmap('Greys'),
                ax = axi)
    sns.despine(right=False, top=False, left=False)
    # axi.set_yticks(xticks)
    # axi.set_yticklabels(xticks)
    # axi.set_xticks(xticks)
    # axi.set_xticklabels(xticks)
    axi.set_aspect('equal')
    
    plt.show()
    
    

def DP1(G1,G2,adj1,adj2):
    n1 = G1.shape[0]
    n2 = G2.shape[0]
    
    d1 = torch.sum(G1,dim=1).long()
    d2 = torch.sum(G2,dim=1).long()
    
    X1 = torch.zeros(n1,n1)
    X2 = torch.zeros(n2,n2)
    
    # plotS(-torch.abs(d1.view(n1,1)-d2.view(1,n2)))
    Result = []
    W = torch.zeros(n1,n2)
    for i in range(n1):
        for j in range(n2):
            W[i,j] = -wasserstein_distance(d1[adj1[i]],d2[adj2[j]])
            # W[j,i] = W[i,j]
    Result.append(W)
    
    # S = [(row[i],col[i],W[row[i],col[i]]) for i in range(n1)]
    # Ss = sorted(S,key=lambda x:-x[2])
    # for i in range(n1):
    #     print(Ss[i])
    for si in range(3):
#         S1 = sinkhorn(W,5)
        S1 = sinkhorn(W*(2+si*0.4),5)
        row, col = linear_sum_assignment(-S1.detach().numpy())
        S = torch.zeros(n1,n2)
        S[row,col] = 1
        S *= S1
        W = torch.mm(G1,(torch.mm(S,G2)))
        Result.append(W)
        
    return Result

def DF1(G1,G2,adj1,adj2):
    n1 = G1.shape[0]
    n2 = G2.shape[0]
    
    d1 = torch.sum(G1,dim=1)
    d2 = torch.sum(G2,dim=1)
    d_mean = torch.mean(d1)
    
    D_diff = sinkhorn(-torch.abs(d1.view(n1,1)-d2.view(1,n2)),5).detach().numpy()
    
    W = torch.zeros(n1,n2)
    for i in range(n1):
        for j in range(n2):
            small_W = D_diff[adj1[i],:][:,adj2[j]]
            row, col = linear_sum_assignment(-small_W)
            W[i,j] = float(sum(small_W[row,col]))
    
    # S = [(row[i],col[i],W[row[i],col[i]]) for i in range(n1)]
    # Ss = sorted(S,key=lambda x:-x[2])
    # for i in range(n1):
    #     print(Ss[i])
    
#     r1 = [] 
#     w1 = []
    
#     for i in range(n1):
#         if row[i]==col[i]:
#             r1.append(float(W[row[i],col[i]]))
#         else:
#             w1.append(float(W[row[i],col[i]]))
#     fig = plt.figure(figsize=(9, 6))

#     ax = plt.subplot(111)
#     ax.hist([r1,w1],bins=10,label = ['T','F'])
    
    
#     plt.legend()
#     plt.show()
    return W

def DA1(G1,G2,adj1,adj2,t):
    n1 = G1.shape[0]
    n2 = G2.shape[0]
    
    d1 = torch.sum(G1,dim=1)
    d2 = torch.sum(G2,dim=1)
    d_mean = torch.mean(d1)
    
    D_diff = -torch.abs(d1.view(n1,1)-d2.view(1,n2))
    D_diff[D_diff>=-t] = 1
    D_diff[D_diff< -t] = 0
                        
    W = torch.mm(G1,(torch.mm(D_diff,G2)))                    
    return W

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
    
    truth = torch.randperm(n)
    # truth = n-1-torch.arange(n)
    # truth = torch.arange(n)
        
    G1 = G1[truth,:][:,truth]
    
    one_to_n = torch.arange(n)
    y = torch.stack((one_to_n, truth))
    
    adj1 = []
    adj2 = []
        
    for i in range(n):
        adj1.append(torch.nonzero(G1[i,:], as_tuple=True)[0].tolist())
    for i in range(n):
        adj2.append(torch.nonzero(G2[i,:], as_tuple=True)[0].tolist())
        
    return (G1, G2, adj1, adj2, y)



def run(n,p,S,Itera):
    
    dp = torch.zeros(Itera,len(S))
    dp2 = torch.zeros(Itera,len(S))
    dp3 = torch.zeros(Itera,len(S))
    dp4 = torch.zeros(Itera,len(S))
    for si, s in enumerate(S):
        for itera in range(Itera):
            
            G1, G2, adj1, adj2, y = Correlated_ER_Graph(n,p,s)
            
            W = DP1(G1,G2,adj1,adj2)
            row, col = linear_sum_assignment(-W[0].detach().numpy())
            result = torch.tensor(col)
            dp[itera,si] = sum((result[y[0]]==y[1]).float())/n
            
            
            row, col = linear_sum_assignment(-W[1].detach().numpy())
            result = torch.tensor(col)
            dp2[itera,si] = sum((result[y[0]]==y[1]).float())/n
            
            row, col = linear_sum_assignment(-W[2].detach().numpy())
            result = torch.tensor(col)
            dp3[itera,si] = sum((result[y[0]]==y[1]).float())/n
            
            row, col = linear_sum_assignment(-W[3].detach().numpy())
            result = torch.tensor(col)
            dp4[itera,si] = sum((result[y[0]]==y[1]).float())/n
            
    
            
    _,dp = torch.std_mean(dp,dim=0,unbiased=False)
    _,dp2 = torch.std_mean(dp2,dim=0,unbiased=False)
    _,dp3 = torch.std_mean(dp3,dim=0,unbiased=False)
    _,dp4 = torch.std_mean(dp4,dim=0,unbiased=False)

    
    S = ', '.join(str(round(i, 4)) for i in S)
    dp = ', '.join(str(round(i, 4)) for i in (dp).tolist())
    dp2 = ', '.join(str(round(i, 4)) for i in (dp2).tolist())
    dp3 = ', '.join(str(round(i, 4)) for i in (dp3).tolist())
    dp4 = ', '.join(str(round(i, 4)) for i in (dp4).tolist())
    
    torch.set_printoptions(precision=4)
    print(f'Parameters: n={n}, p={p}')
    print('Accuracy')
    print('s = '.ljust(10), S)
    print('dp = '.ljust(10), dp)
    print('dp2 = '.ljust(10), dp2)
    print('dp3 = '.ljust(10), dp3)
    print('dp4 = '.ljust(10), dp4)


n = 100
p = 0.2
S = [0.99, 0.97, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]
# S = [0.85]
Itera = 10
start_time = time.time()
run(n, p, S, Itera)
print("--- %s seconds ---" % (time.time() - start_time))

# print('-----------------------------------------------')

# n = 1000
# p = 0.1
# S = [0.99,0.97, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]
# Itera = 1
# run(n,p,S,Itera)
