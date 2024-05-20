import sys
import copy
import os
import os.path as osp
import numpy as np
from scipy import sparse as sp
from scipy.optimize import linear_sum_assignment
import random
import time
import argparse
import torch
import torch_geometric.transforms as T
from multiprocessing import Pool
from itertools import product
import time

from models.PairGNN import GNNGM1 as GNNGM

from GMAlgorithms import facebookGraph, FAQ, PATH, Grampa
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
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = GNNGM(args.num_layers, args.gnn_layers,args.hid).to(device)

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
        adj1.append(torch.nonzero(G1[i,:], as_tuple=True)[0])
    for i in range(n):
        adj2.append(torch.nonzero(G2[i,:], as_tuple=True)[0])
        
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
        
        
        total_correct += correct
        total_node += num_nodes
        total_c += c
    return total_correct/total_node, total_c/total_node

def run(numgraphs,S,Itera):
    
    Facebook_Filepath = "./data/facebook100"
    filedirs = os.listdir(Facebook_Filepath)
    
    gnn = torch.zeros(len(S))
    grampa = torch.zeros(len(S))
    sgm = torch.zeros(len(S))
    ntma = torch.zeros(len(S))
    graphi = 0
    for realpath in filedirs[:numgraphs]:
        print(realpath)
        if os.path.splitext(realpath)[1]=='.mat':
            
            for si, s in enumerate(S):
                for itera in range(Itera):
            
                    G1, G2, adj1,adj2, y = facebookGraph(Facebook_Filepath+'/'+realpath,s,0.1,0)
                    n = G1.shape[0]
                    print(n,torch.mean(G1))
                    gnn[si] += test([(G1, G2, adj1, adj2, y)])[0]
            
                    result = FAQ(G1,G2)
                    sgm[si] += sum((result[y[0]]==y[1]).float())/n
            
                    W = ntma_v.match(G1,G2,adj1,adj2)
                    
                    row, col = linear_sum_assignment(-W[0].detach().numpy())
                    result = torch.tensor(col)
                    ntma[itera,si] = sum((result[y[0]]==y[1]).float())/n 
            
                    result = Grampa(G1,G2,0.2)
                    grampa[si] += sum((result[y[0]]==y[1]).float())/n        
        graphi +=1
        
    S = ', '.join(str(round(i, 4)) for i in S)
    gnn = ', '.join(str(round(i, 4)) for i in (gnn/graphi/Itera).tolist())
    grampa = ', '.join(str(round(i, 4)) for i in (grampa/graphi/Itera).tolist())
    sgm = ', '.join(str(round(i, 4)) for i in (sgm/graphi/Itera).tolist())
    ntma = ', '.join(str(round(i, 4)) for i in (ntma/graphi/Itera).tolist())
    with open('TestFB.txt', 'a') as f:
        f.write('s = ['+S+']\n')
        f.write('GNN = ['+gnn+']\n')
        f.write('grampa = ['+grampa+']\n')
        f.write('sgm  = ['+sgm+']\n' )
        f.write('ntma = ['+ntma+']\n')
        f.write('-----------------------------------------\n')

    torch.set_printoptions(precision=4)
    print('Accuracy')
    print('s = '.ljust(10), S)
    print('LGM-GNN = '.ljust(10),gnn)
    print('grampa = '.ljust(10), grampa)
    print('sgm = '.ljust(10), sgm)
    print('ntma = '.ljust(10), ntma)


path = "./model/Seedless-prod-8.pth"
model.load_state_dict(torch.load(path))
open('TestFB.txt', 'w')

numgraphs = 1
S = [0.95]
Itera = 1
run(numgraphs,S,Itera)

print('-----------------------------------------------')
