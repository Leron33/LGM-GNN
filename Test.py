import os
import os.path as osp
import math
import random
import argparse
import copy
import time
from multiprocessing import Pool, freeze_support
import numpy as np
from scipy import sparse as sp
from scipy.optimize import linear_sum_assignment

import torch

print(torch.rand(2,2))