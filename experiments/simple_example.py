#!/usr/bin/env python

"""
    simple_example.py
"""

import sys
import numpy as np
from time import time
from tqdm import tqdm, trange

import torch
from torch import nn

from rsub import *
from matplotlib import pyplot as plt

from hoof import dataset
from hoof.models import ALPACA, rks_regression
from hoof.helpers import set_seeds, to_numpy, list2tensors, tensors2list, set_lr

torch.set_default_tensor_type('torch.DoubleTensor')
torch.set_num_threads(1)
set_seed(123)

# --
# Dataset

num_problems = 30

dataset_name  = 'SinusoidDataset'
dataset_cls   = getattr(dataset, dataset_name)
train_dataset = dataset_cls()
valid_dataset = dataset_cls()

def make_dataset(dataset, num_problems):
    problems, fns = [], []
    for _ in range(num_problems):
        x_s, y_s, _, _, fn = dataset.sample_one(support_size=10, query_size=0)
        problems.append([
            torch.Tensor(x_s),
            torch.Tensor(y_s),
        ])
        fns.append(fn)
    
    return problems, fns

train_problems, train_fns = make_dataset(train_dataset, num_problems)
valid_problems, valid_fns = make_dataset(valid_dataset, 10 * num_problems)

# --
# Train

model = ALPACA(input_dim=1, output_dim=1, sig_eps=0.01, hidden_dim=128, activation='tanh').cuda()

train_history = []
lrs = [1e-4, 1e-5]
opt = torch.optim.Adam(model.parameters(), lr=lrs[0])

train_kwargs = {"batch_size" : 64, "support_size" : 10, "query_size" : 100, "num_samples" : 30000}

for lr in lrs:
    set_lr(opt, lr)
    
    train_history += model.train(dataset=train_problems, opt=opt, **train_kwargs)
    
    _ = plt.plot(train_history, c='red', label='train')
    _ = plt.yscale('log')
    _ = plt.grid()
    _ = plt.legend()
    show_plot()


valid_history = model.valid(dataset=valid_dataset, **train_kwargs)

print('final_train_loss=%f' % np.mean(train_history[-100:]), file=sys.stderr)
print('final_valid_loss=%f' % np.mean(valid_history[-100:]), file=sys.stderr)

# --
# Plot example

np.random.seed(456)
valid_dataset.set_seed(123)
fig, ax = plt.subplots(3, 3)

for i in range(3):
    for j in range(3):
        x_s, y_s, _, _, fn = valid_dataset.sample_one(support_size=5, query_size=0)
        x_grid = np.linspace(*valid_dataset.x_range, 1000).reshape(-1, 1)
        y_grid = fn(x_grid)
        
        inp = list2tensors((x_s, y_s, x_grid), cuda=model.is_cuda)
        mu, sig, _ = model(*inp)
        mu, sig = tensors2list((mu, sig), squeeze=True)
        
        _ = ax[i,j].plot(x_grid, y_grid, c='black', alpha=0.25)
        _ = ax[i,j].plot(x_grid, mu)
        _ = ax[i,j].plot(x_grid, y_prior.squeeze())
        # _ = ax[i,j].plot(x_grid, rks_regression(x_s, y_s, x_grid, n_components=50, gamma=0.25), c='orange', alpha=0.75)
        _ = ax[i,j].fill_between(x_grid.squeeze(), mu - 1.96 * np.sqrt(sig), mu + 1.96 * np.sqrt(sig), alpha=0.2)
        _ = ax[i,j].scatter(x_s, y_s, c='red')
        _ = ax[i,j].set_xlim(*valid_dataset.x_range)

show_plot()

