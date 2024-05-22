# -*- coding: utf-8 -*-
"""
Created on Wed May 22 00:57:07 2024

@author: liren
"""

import sys
import os
import numpy as np
from scipy import sparse as sp
from scipy.optimize import linear_sum_assignment
import random

import matplotlib.pyplot as plt
import seaborn as sns

import argparse
import torch
import torch_geometric.transforms as T
from multiprocessing import Pool
from itertools import product

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
        D_diff = d1+d2+torch.abs(d1-d2)
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



def plotS(S):
    fig = plt.figure(figsize=(6, 6))
    axi = plt.subplot(1, 1, 1)

    S_min = torch.min(S)
    S_max = torch.max(S)
    S = sinkhorn((S - S_min) / (S_max - S_min),5)

    S = torch.rot90(S, 3).detach().numpy()
    print(S)
    sns.heatmap(data=list(S), cbar=False, cmap=plt.get_cmap('Greys'), ax=axi)
    sns.despine(right=False, top=False, left=False)
    # axi.set_yticks(xticks)
    # axi.set_yticklabels(xticks)
    # axi.set_xticks(xticks)
    # axi.set_xticklabels(xticks)
    axi.set_aspect('equal')

    plt.show()

def run(n,p,S,Itera):
    
    gnn = torch.zeros(Itera,len(S))
    ntma = torch.zeros(Itera,len(S))
    ntma1 = torch.zeros(Itera,len(S))
    ntma2 = torch.zeros(Itera,len(S))
    ntma3 = torch.zeros(Itera,len(S))
    ntma4 = torch.zeros(Itera,len(S))
    dp = torch.zeros(Itera,len(S))

    for si, s in enumerate(S):
        for itera in range(Itera):
            
            G1, G2, adj1, adj2, y = Correlated_ER_Graph(n,p,s)
            
            # gnn[itera,si] = test([(G1, G2, adj1, adj2, y)])[0]
            
            # result = FAQ(G1,G2,torch.tensor([[],[]]).long())
            # sgm[itera,si] = sum((result[y[0]]==y[1]).float())/n
            
            W = ntma_v.match(G1,G2,adj1,adj2)
            
            for i in range(4):
                S = W[i]
                fig = plt.figure(figsize=(6, 6))
                axi = plt.subplot(1, 1, 1)

                S_min = torch.min(S)
                S_max = torch.max(S)
                S = sinkhorn((S - S_min) / (S_max - S_min),5)
                S = torch.rot90(S, 3).detach().numpy()
#                 print(S)
                sns.heatmap(data=list(S), cbar=False, cmap=plt.get_cmap('Greys'), ax=axi)
                sns.despine(right=False, top=False, left=False)
                # axi.set_yticks(xticks)
                # axi.set_yticklabels(xticks)
                # axi.set_xticks(xticks)
                # axi.set_xticklabels(xticks)
                axi.set_aspect('equal')
#                 plt.savefig('./figure/ER_sparse_NTMA'+'{}'.format(i)+'.eps',bbox_inches ='tight')
                plt.show()


n = 100
p = 0.05
# S = [0.99,0.97, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]
S = [0.9]
Itera = 1
run(n,p,S,Itera)

# print('-----------------------------------------------')

n = 100
p = 0.2
# S = [0.99,0.97, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]
S = [0.9]
Itera = 1
run(n,p,S,Itera)