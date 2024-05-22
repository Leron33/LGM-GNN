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
from models.PairGNN import GNNGM1 as GNNGM
# from GMAlgorithms import SGM as FAQ, PATH, Grampa, NTMA, DP,DP1
torch.set_printoptions(precision=4)

parser = argparse.ArgumentParser()
parser.add_argument('--hid', type=int, default=8)
parser.add_argument('--num_layers', type=int, default=8)
parser.add_argument('--gnn_layers', type=int, default=2)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--gamma', type=float, default=0.8)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--runs', type=int, default=20)

args, unknown = parser.parse_known_args()
device = 'cpu'#'cuda' if torch.cuda.is_available() else 'cpu'

model = GNNGM(args.num_layers, args.gnn_layers,args.hid).to(device)

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
    axi = plt.subplot(1, 1, 1)

#     S_min = torch.min(S)
#     S_max = torch.max(S)
#     S = (S - S_min) / (S_max - S_min)

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
        adj1.append(torch.nonzero(G1[i, :], as_tuple=True)[0].tolist())
    for i in range(n):
        adj2.append(torch.nonzero(G2[i, :], as_tuple=True)[0].tolist())
        
    return (G1, G2, adj1, adj2, y)

@torch.no_grad()
def test(test_dataset):
    model.eval()

    total_correct = 0
    total_node = 0
    total_c = 0
    for data in test_dataset:
        
        G1 = data[0]
        G2 = data[1]
        adj1 = data[2]
        adj2 = data[3]
        Y = data[4]
        num_nodes = G1.shape[0]
        
        W,L= model(G1, G2, adj1, adj2)
        
        correct = model.acc(W, Y)
        c = 0# model.acc(L[0], Y)
        
        
        # row,col = linear_sum_assignment(-L[0].detach().numpy())
        # print(c,col)
        # row,col = linear_sum_assignment(-W.detach().numpy())
        # print(correct,col)
        # print(W.max(),W)
        # print(G1.to_sparse().indices())
        # print(num_nodes-1-G2.to_sparse().indices())
        # sys.exit(0)
        
        total_correct += correct
        total_node += num_nodes
        total_c += c
    return total_correct/total_node, total_c/total_node

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
            
            O,W = model(G1, G2, adj1, adj2)
            
            for i in range(4):
                S = W[i+1]
                fig = plt.figure(figsize=(6, 6))
                axi = plt.subplot(1, 1, 1)

#                 S_min = torch.min(S)
                S_max = torch.max(S)
#                 S = (S - S_min) / (S_max - S_min)
                if i ==0:
                    S_max *=1.2
                S = torch.rot90(S, 3).detach().numpy()
#                 print(S)
                sns.heatmap(data=list(S),vmax = S_max, vmin =0, cbar=False, cmap=plt.get_cmap('Greys'), ax=axi)
                sns.despine(right=False, top=False, left=False)
                # axi.set_yticks(xticks)
                # axi.set_yticklabels(xticks)
                # axi.set_xticks(xticks)
                # axi.set_xticklabels(xticks)
                axi.set_aspect('equal')
#                 plt.savefig('./figure/ER_sparse_PWGNN'+'{}'.format(i)+'.eps',bbox_inches ='tight')
                plt.show()
            
            

path = "./model/Seedless-prod-8.pth"
model.load_state_dict(torch.load(path))
# open('TestER2.txt', 'w')

n = 100
p = 0.05
# S = [0.99, 0.97, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]
S = [0.9]
Itera = 1
run(n,p,S,Itera)

# print('-----------------------------------------------')

n = 100
p = 0.2
# S = [0.99, 0.97, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]
S = [0.9]
Itera = 1
run(n,p,S,Itera)