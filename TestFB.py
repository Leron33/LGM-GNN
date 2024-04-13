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

def run(numgraphs,S,Itera):
    
    Facebook_Filepath = "./data/facebook100"
    filedirs = os.listdir(Facebook_Filepath)
    
    gnn = torch.zeros(len(S))
    grampa = torch.zeros(len(S))
    sgm = torch.zeros(len(S))
    faqd = torch.zeros(len(S))
    graphi = 0
    for realpath in filedirs[:numgraphs]:
        print(realpath)
        if os.path.splitext(realpath)[1]=='.mat':
            
            for si, s in enumerate(S):
                for itera in range(Itera):
            
                    G1, G2, adj1,adj2, y = facebookGraph(Facebook_Filepath+'/'+realpath,s,0.25,0)
                    n = G1.shape[0]
                    print(n,torch.mean(G1))
                    gnn[si] += test([(G1, G2, adj1, adj2, y)])[0]
            
                    result = FAQ(G1,G2)
                    sgm[si] += sum((result[y[0]]==y[1]).float())/n
            
                    result = PATH(G1,G2)
                    faqd[si] += sum((result[y[0]]==y[1]).float())/n
            
                    result = Grampa(G1,G2,0.2)
                    grampa[si] += sum((result[y[0]]==y[1]).float())/n        
        graphi +=1
        
    S = ', '.join(str(round(i, 4)) for i in S)
    gnn = ', '.join(str(round(i, 4)) for i in (gnn/graphi/Itera).tolist())
    grampa = ', '.join(str(round(i, 4)) for i in (grampa/graphi/Itera).tolist())
    sgm = ', '.join(str(round(i, 4)) for i in (sgm/graphi/Itera).tolist())
    faqd = ', '.join(str(round(i, 4)) for i in (faqd/graphi/Itera).tolist())
    with open('TestFB.txt', 'a') as f:
        f.write('s = ['+S+']\n')
        f.write('GNN = ['+gnn+']\n')
        f.write('grampa = ['+grampa+']\n')
        f.write('sgm  = ['+sgm+']\n' )
        f.write('faqd = ['+faqd+']\n')
        f.write('-----------------------------------------\n')

    torch.set_printoptions(precision=4)
    print('Accuracy')
    print('s = '.ljust(10), S)
    print('GNN = '.ljust(10),gnn)
    print('grampa = '.ljust(10), grampa)
    print('sgm = '.ljust(10), sgm)
    print('faqd = '.ljust(10), faqd)


path = "./model/Seedless-prod-8.pth"
model.load_state_dict(torch.load(path))
open('TestFB.txt', 'w')

numgraphs = 10
S = [0.95]
Itera = 1
run(numgraphs,S,Itera)

print('-----------------------------------------------')
