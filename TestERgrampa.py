# -*- coding: utf-8 -*-
"""
Created on Tue May 21 23:10:57 2024

@author: liren
"""

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
from GMAlgorithms import Grampa

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
    
    grampa = torch.zeros(Itera,len(S))
    
    for si, s in enumerate(S):
        for itera in range(Itera):
            
            G1, G2, adj1, adj2, y = Correlated_ER_Graph(n,p,s)
            
            result = Grampa(G1,G2,0.2)
            grampa[itera,si] = sum((result[y[0]]==y[1]).float())/n
            
    
            
    _,grampa = torch.std_mean(grampa,dim=0,unbiased=False)

    
    S = ', '.join(str(round(i, 4)) for i in S)
    grampa = ', '.join(str(round(i, 4)) for i in (grampa).tolist())
    
    torch.set_printoptions(precision=4)
    print(f'Parameters: n={n}, p={p}')
    print('Accuracy')
    print('s = '.ljust(10), S)
    print('grampa = '.ljust(10), grampa)


n = 1000
p = 0.01
S = [0.99, 0.97, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]
# S = [0.85]
Itera = 10
start_time = time.time()
run(n, p, S, Itera)
print("--- %s seconds ---" % (time.time() - start_time))

print('-----------------------------------------------')

n = 1000
p = 0.1
S = [0.99, 0.97, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]
# S = [0.85]
Itera = 10
start_time = time.time()
run(n, p, S, Itera)
print("--- %s seconds ---" % (time.time() - start_time))
