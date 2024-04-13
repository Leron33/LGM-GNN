import os
import os.path as osp
import matplotlib.pyplot as plt
import math
import random
import argparse
import copy

import numpy as np
from scipy import sparse as sp
from scipy.optimize import linear_sum_assignment
from multiprocessing import Pool, freeze_support
import time

import torch
from GMAlgorithms import FAQ,PATH,Grampa
from models.PairGNN import GNNGM1 as GNNGM

# from GMAlgorithms import SynGraph, facebookGraph

###################################################################################

torch.set_printoptions(precision=4)

parser = argparse.ArgumentParser()
parser.add_argument('--hid', type=int, default=8)
parser.add_argument('--num_layers', type=int, default=8)
parser.add_argument('--gnn_layers', type=int, default=2)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--gamma', type=float, default=1.0)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--runs', type=int, default=20)
num_hops = 3

args, unknown = parser.parse_known_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device ='cpu'

model = GNNGM(args.num_layers, args.gnn_layers,args.hid).to(device)

#################################################################################


def triangulatoin2adj(realpath):
    f = open(realpath, 'r')
    f.readline() # OFF
    num_views, num_groups, num_edges = map(int, f.readline().split())
    view_data = []
    for view_id in range(num_views):
        view_data.append(list(map(float, f.readline().split())))    
    group_data = []
    for group_id in range(num_groups):
        group_data.append(list(map(int, f.readline().split()[1:])))
    
    f.close()
    
    
    adj = torch.zeros(num_views,num_views)
    for face in group_data:
        for k in range(3):
            kk = (k+1)%3
            adj[face[k]-1,face[kk]-1] = 1
    adj = torch.max(adj,adj.T)
    return adj

def merge_gt(path1,path2):
    f = open(path1, 'r')
    gt1 = []
    while True:
        line = f.readline()
        if not line:    
            break
        else:
            gt1.append(list(map(int, line.split())))
    f.close()
    
    f = open(path2, 'r')
    gt2 = []
    while True:
        line = f.readline()
        if not line:   
            break
        else:
            gt2.append(list(map(int, line.split())))
    f.close()

    maxnode = max(max(max(gt1)),max(max(gt2)))

    bin1=torch.zeros(maxnode)
    bin2=torch.zeros(maxnode)
    set1 = set()
    set2 = set()
    for maps in gt1:
        bin1[maps[1]-1]=maps[0]-1
        set1.add(maps[1]-1)
    for maps in gt2:
        bin2[maps[1]-1]=maps[0]-1
        set2.add(maps[1]-1)
    joint = set1.intersection(set2)
    gt = torch.zeros(2,len(joint))
    ni = 0
    for i in joint:
        gt[0][ni] = bin1[i]
        gt[1][ni] = bin2[i]
        ni += 1
    return gt.long()

def ShrecGraph(realpath,i,j):
    path = realpath +str(i)+'.off'    
    G1 = triangulatoin2adj(path)

    path = realpath +str(j)+'.off'
    G2 = triangulatoin2adj(path)
    
    path1 = realpath +str(i)+'_ref.txt'    
    path2 = realpath +str(j)+'_ref.txt'  
    truth = merge_gt(path1,path2)
    n1 = G1.shape[0]
    n2 = G2.shape[0]
    
    adj1 = []
    adj2 = []
        
    for i in range(n1):
        adj1.append(torch.nonzero(G1[i,:], as_tuple=True)[0].tolist())
    for i in range(n2):
        adj2.append(torch.nonzero(G2[i,:], as_tuple=True)[0].tolist())
    
    return (G1, G2, adj1, adj2, truth)

###################################################################################


@torch.no_grad()
def test(test_dataset):
    model.eval()

    total_correct = 0
    total_acc = 0
    num_test = 0
    
    for data in test_dataset:
        
        G1 = data[0]
        G2 = data[1]
        adj1 = data[2]
        adj2 = data[3]
        Y = data[4]
        n1 = G1.shape[0]
        
        W, L= model(G1, G2, adj1, adj2,20)
        
        correct = model.acc(W, Y)
        total_acc += correct/n1
        num_test += 1
    return total_acc/num_test


def run():
    
    Shrec_Filepath = "./data/low_resolution/kid"
    
    datasets = []
    total_acc= 0
    num_test = 0
    for i in range(16,25):
        for j in range(i+1,25):
            G1, G2, adj1, adj2, y = ShrecGraph(Shrec_Filepath,i,j)            
            n1 = G1.shape[0]
            
            
                       
            result = FAQ(G1,G2)
            sgm = sum((result[y[0]]==y[1]).float())/n1
            
            result = PATH(G1,G2)
            faqd = sum((result[y[0]]==y[1]).float())/n1
            
            result = Grampa(G1,G2,0.2)
            grampa = sum((result[y[0]]==y[1]).float())/n1
            
            
            
            #total_acc += test_acc
            num_test += 1
    #acc = total_acc/num_test
    #print("Acc: ", acc)
    print('grampa = '.ljust(10), grampa)
    print('sgm = '.ljust(10), sgm)
    print('faqd = '.ljust(10), faqd)        
            

    return
    
 
if __name__ == "__main__": 
    freeze_support()
    path = "./model/Seedless-prod-8.pth"
    model.load_state_dict(torch.load(path))
    start_time = time.perf_counter()
    run()
    finish_time = time.perf_counter()
    print("Program finished in {} seconds".format(finish_time-start_time))   