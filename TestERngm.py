# -*- coding: utf-8 -*-
"""
Created on Tue May 21 23:51:34 2024

@author: liren
"""

import os
import os.path as osp
import math
import random
import argparse
import copy

import numpy as np
from scipy import sparse as sp
from scipy.optimize import linear_sum_assignment

import torch

from multiprocessing import Pool, freeze_support
import time

from models.NGM import NGM as GNNGM
# from GMAlgorithms import SynGraph, facebookGraph

###################################################################################

torch.set_printoptions(precision=4)

parser = argparse.ArgumentParser()
parser.add_argument('--hid', type=int, default=8)
parser.add_argument('--num_layers', type=int, default=8)
parser.add_argument('--gnn_layers', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--n', type=int, default=100)
parser.add_argument('--p', type=float, default=0.2)
parser.add_argument('--s', type=float, default=0.8)
torch.manual_seed(42)

args, unknown = parser.parse_known_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = "cpu"
model = GNNGM(args.num_layers, args.hid).to(device)

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

@torch.no_grad()
def test(test_dataset):
    model.eval()

    total_correct =0
    total_c = 0
    total_node = 0
    for data in test_dataset:
        
        G1 = data[0]
        G2 = data[1]
        adj1 = data[2]
        adj2 = data[3]
        Y = data[4]
        num_nodes = G1.shape[0]
        
        W,L= model(G1, G2)
        
        correct = model.acc(W, Y)
        total_correct+= correct
        total_node += num_nodes
    return total_correct/total_node, total_c/total_node

def run(n,p,S,Itera):
    
    gnn = torch.zeros(len(S))

    for si, s in enumerate(S):
        for itera in range(Itera):
            
            G1, G2, adj1, adj2, y = Correlated_ER_Graph(n,p,s)
            
            gnn[si] += test([(G1, G2, adj1, adj2, y)])[0]
            
           

    gnn = gnn[:]/Itera
   
    
    #S = ', '.join(str(round(i, 4)) for i in S)
    #gnn = ', '.join(str(round(i, 4)) for i in (gnn).tolist())
    
    # with open('TestER2.txt', 'a') as f:
    #     f.write('s = ['+S+']\n')
    #     f.write('GNN = ['+gnn+']\n')
    #     f.write('grampa = ['+grampa+']\n')
    #     f.write('sgm  = ['+sgm+']\n' )
    #     f.write('faqd = ['+ntma+']\n')
    #     f.write('dp = ['+dp+']\n')
    #     f.write('-----------------------------------------\n')

    torch.set_printoptions(precision=4)
    print('Accuracy')
    print('s = '.ljust(10), S)
    print('NGM = '.ljust(10),gnn)


path = "./model/NGM.pth"
model.load_state_dict(torch.load(path))


n = 1000
p = 0.01
S = [0.99,0.97, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]
Itera = 10
print(f'n={n}, p={p}, s= ', S)
start_time = time.time()
run(n, p, S, Itera)
print("--- %s seconds ---" % (time.time() - start_time))

print('-----------------------------------------------')

n = 1000
p = 0.1
S = [0.99,0.97, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]
Itera = 10
print(f'n={n}, p={p}, s= ', S)
start_time = time.time()
run(n, p, S, Itera)
print("--- %s seconds ---" % (time.time() - start_time))
