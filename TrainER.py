import os
import sys
import os.path as osp
import matplotlib.pyplot as plt
import math
import random
import argparse
import copy

import numpy as np
from scipy import sparse as sp
from scipy.optimize import linear_sum_assignment

import torch

from models.PairGNN import GNNGM1 as GNNGM

# from GMAlgorithms import SynGraph, facebookGraph

###################################################################################

torch.set_printoptions(precision=8)

parser = argparse.ArgumentParser()
parser.add_argument('--hid', type=int, default=8)
parser.add_argument('--num_layers', type=int, default=8)
parser.add_argument('--gnn_layers', type=int, default=2)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--runs', type=int, default=20)

args, unknown = parser.parse_known_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

model = GNNGM(args.num_layers, args.gnn_layers, args.hid).to(device)

#################################################################################


def Correlated_ER_Graph(n, p, s):

    a = (torch.rand(n, n) < p).float()
    a = torch.triu(a, diagonal=1)
    G0 = a + a.T

    sample = (torch.rand(n, n) < s).float()
    sample = torch.triu(sample, 1)
    sample = sample + sample.T
    G1 = G0 * sample

    sample = (torch.rand(n, n) < s).float()
    sample = torch.triu(sample, 1)
    sample = sample + sample.T
    G2 = G0 * sample

    truth = torch.randperm(n)
    #truth = n-1-torch.arange(n)
    #truth = torch.arange(n)

    G1 = G1[truth, :][:, truth]

    one_to_n = torch.arange(n)
    y = torch.stack((one_to_n, truth))

    adj1 = []
    adj2 = []

    for i in range(n):
        adj1.append(torch.nonzero(G1[i, :], as_tuple=True)[0].tolist())
    for i in range(n):
        adj2.append(torch.nonzero(G2[i, :], as_tuple=True)[0].tolist())

    return (G1, G2, adj1, adj2, y)


###################################################################################

def train(train_dataset, optimizer):
    model.train()

    total_loss = 0
    num_examples = 0

    for i in range(0, len(train_dataset), args.batch_size):

        batch = train_dataset[i:i + args.batch_size]
        optimizer.zero_grad()
        batch_loss = 0

        for data in batch:

            G1 = data[0]
            G2 = data[1]
            adj1 = data[2]
            adj2 = data[3]
            Y = data[4]
            n1 = G1.shape[0]

            W, L = model(G1, G2, adj1, adj2)

            loss = model.loss(L, Y)
            batch_loss += loss
            total_loss += loss
            num_examples += 1

        batch_loss.backward()
        optimizer.step()

    return total_loss.item() / num_examples


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

        W, L = model(G1, G2, adj1, adj2)
        #print(W)
        #sys.exit(0)
        correct = model.acc(W, Y)
        total_acc += correct / n1
        num_test += 1
    return total_acc / num_test


def run(train_dataset, test_dataset):

    path = "./model/Seedless-prod-8-50.pth"
#     model.load_state_dict(torch.load(path))
    model.reset_parameters()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=10,
                                                gamma=args.gamma)

    for epoch in range(1, 1 + args.epochs):
        loss = train(train_dataset, optimizer)
        scheduler.step()

        if epoch % 5 == 0:
            train_acc = test(train_dataset)
            test_acc = test(test_dataset)
            print(
                f'epoch {epoch:03d}: Loss: {loss:.8f}, Training Acc: {train_acc:.4f}, Testing Acc: {test_acc:.4f}'
            )


#     path = "./model/GNNseed-42.pth"
    torch.save(model.state_dict(), path)

    return train_acc

if __name__ == '__main__':
    print('Preparing training data...')
    train_dataset = []
    test_dataset = []
    n = 100
    p = n**(-1 / 3)
    s = 0.85
    graph_para = [(n, p, s)]
    numgraphs = 100
    print(graph_para)
    for n, p, s in graph_para:
        for _ in range(50):
            train_dataset.append(Correlated_ER_Graph(n, p, s))
        for _ in range(numgraphs):
            test_dataset.append(Correlated_ER_Graph(n, p, s))

    print('Done!')
    run(train_dataset, test_dataset)