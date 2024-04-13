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

from models.PairGNN import GNNGM1 as GNNGM
# from GMAlgorithms import SynGraph, facebookGraph

###################################################################################

torch.set_printoptions(precision=4)

parser = argparse.ArgumentParser()
parser.add_argument('--hid', type=int, default=8)
parser.add_argument('--num_layers', type=int, default=8)
parser.add_argument('--gnn_layers', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--n', type=int, default=100)
parser.add_argument('--p', type=float, default=0.22)
parser.add_argument('--s', type=float, default=0.85)
num_hops = 3

args, unknown = parser.parse_known_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

model = GNNGM(args.num_layers, args.gnn_layers, args.hid).to(device)
torch.set_default_device(device)
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
    # truth = n-1-torch.arange(n)
    # truth = torch.arange(n)

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

        W, L = model(G1, G2, adj1, adj2, 6)

        correct = model.acc(W, Y)
        total_acc += correct / n1
        num_test += 1
    return total_acc / num_test


def run(n, p, s, Itera):
    datasets = []
    total_acc = torch.zeros((Itera, 1))
    for i in range(Itera):

        test_acc = test([Correlated_ER_Graph(n, p, s)])
        total_acc[i] = test_acc
    std, acc = torch.std_mean(total_acc, dim=0, unbiased=False)
    with open('TestER.txt', 'a') as f:
        f.write(f'n={n}, p={p}, s={s}\n')
        f.write(f'GMN = {acc}\n')
    f.close()

    print("Acc: ", acc, "std: ", std)
    return


if __name__ == "__main__":

    path = "./model/Seedless-prod-8.pth"
    model.load_state_dict(torch.load(path))
    n = 200
    p = n**(-1/3)
    s = 0.8
    print(f'n={n}, p={p}, s={s}')
    print(path)
    Itera = 1

    start_time = time.time()
    run(n, p, s, Itera)
    print("--- %s seconds ---" % (time.time() - start_time))