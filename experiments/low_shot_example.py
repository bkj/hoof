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

torch.set_num_threads(1)
set_seeds(345)

# --
# Dataset

dataset_name = 'SinusoidDataset'

dataset_cls = getattr(dataset, dataset_name)

support_size = 10
query_size   = 100
cache_size   = 10

train_dataset = dataset.CacheDataset(
    dataset=dataset_cls(),
    support_size=support_size,
    query_size=query_size,
    cache_size=cache_size
)

valid_dataset = dataset_cls()

# Plot some examples
# for _ in range(100):
#     x_s, y_s, _, _, fn = train_dataset.sample_one()
#     x_s, y_s = x_s.squeeze(), y_s.squeeze()
    
#     x_grid = np.linspace(*train_dataset.dataset.x_range, 1000).reshape(-1, 1)
#     y_grid = fn(x_grid)
    
#     _ = plt.scatter(x_s, y_s)
#     _ = plt.plot(x_grid, y_grid, alpha=0.25)

# show_plot()

# Compute prior manually
# y_grids = []
# for _ in range(1000):
#     _, _, _, _, fn = valid_dataset.sample_one(support_size=5, query_size=0)
#     x_grid = np.linspace(*valid_dataset.x_range, 1000).reshape(-1, 1)
#     y_grid = fn(x_grid)
#     y_grids.append(y_grid)

# y_prior = np.stack(y_grids).mean(axis=0)
# _ = plt.plot(y_prior.squeeze())
# show_plot()

# --
# Train

model = ALPACA(input_dim=1, output_dim=1, sig_eps=0.2, 
    hidden_dim=128, activation='tanh', learn_priors=False).cuda()

train_history = []
lrs = [1e-4, 1e-4, 1e-4]
opt = torch.optim.Adam(model.parameters(), lr=lrs[0])

train_kwargs = {"batch_size" : 100, "support_size" : support_size, "query_size" : query_size, "num_samples" : 30000}

for lr in lrs:
    set_lr(opt, lr)
    
    train_history += model.train(dataset=train_dataset, opt=opt, **train_kwargs)
    
    _ = plt.plot(train_history, c='red', label='train')
    _ = plt.yscale('log')
    _ = plt.grid()
    _ = plt.legend()
    show_plot()


valid_history = model.valid(dataset=valid_dataset, **train_kwargs)

print('final_train_loss=%f' % np.mean(train_history[-100:]), file=sys.stderr)
print('final_valid_loss=%f' % np.mean(valid_history[-100:]), file=sys.stderr)

_ = plt.plot(train_history, c='red', label='train')
_ = plt.plot(valid_history, c='green', label='valid')
_ = plt.yscale('log')
_ = plt.grid()
_ = plt.legend()
show_plot()


# --
# Plot example

np.random.seed(456)
valid_dataset.set_seed(123)
fig, ax = plt.subplots(3, 3)

for i in range(3):
    for j in range(3):
        x_s, y_s, _, _, fn = valid_dataset.sample_one(support_size=10, query_size=0)
        x_grid = np.linspace(*valid_dataset.x_range, 1000).reshape(-1, 1)
        y_grid = fn(x_grid)
        
        inp = list2tensors((x_s, y_s, x_grid), cuda=model.is_cuda)
        mu, sig, _ = model(*inp)
        mu, sig = tensors2list((mu, sig), squeeze=True)
        
        _ = ax[i,j].plot(x_grid, y_grid, c='black', alpha=0.25)
        _ = ax[i,j].plot(x_grid, mu)
        # _ = ax[i,j].plot(x_grid, y_prior.squeeze())
        # _ = ax[i,j].plot(x_grid, rks_regression(x_s, y_s, x_grid, n_components=50, gamma=0.25), c='orange', alpha=0.75)
        _ = ax[i,j].fill_between(x_grid.squeeze(), mu - 1.96 * np.sqrt(sig), mu + 1.96 * np.sqrt(sig), alpha=0.2)
        _ = ax[i,j].scatter(x_s, y_s, c='red')
        _ = ax[i,j].set_xlim(*valid_dataset.x_range)

show_plot()

