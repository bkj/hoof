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

from hoof.dataset import SinusoidDataset, PowerDataset
from hoof.models import ALPACA
from hoof.helpers import set_seeds, to_numpy, list2tensors, tensors2list

torch.set_num_threads(1)
set_seeds(345)

# --
# Helpers

def mse(act, pred):
    return float(((act - pred) ** 2).mean())


def train(model, opt, dataset, batch_size=10, support_size=5, query_size=5, train_batches=100):
    loss_history = []
    gen = trange(train_batches // batch_size)
    for batch_idx in gen:
        x_support, y_support, x_query, y_query, _ = dataset.sample_batch(
            batch_size=batch_size,
            support_size=support_size, # Could sample this horizon for robustness
            query_size=query_size,     # Could sample this horizon for robustness
        )
        
        (x_support, y_support, x_query, y_query) = \
            list2tensors((x_support, y_support, x_query, y_query), cuda=model.is_cuda)
        
        opt.zero_grad()
        mu, sig, loss = model(x_support, y_support, x_query, y_query)
        loss.backward()
        opt.step()
        
        loss_history.append(mse(mu, y_query))
        
        if not batch_idx % 10:
            gen.set_postfix(loss='%0.8f' % np.mean(loss_history[-10:]))
        
    return loss_history


# --
# Train

dataset = 'power'

if dataset == 'sinusoid':
    train_dataset = SinusoidDataset(noise_std=0.0)
    valid_dataset = SinusoidDataset(noise_std=0.0)
    model = ALPACA(x_dim=1, y_dim=1, sig_eps=0.01, hidden_dim=32, final_dim=32, activation='tanh')
elif dataset == 'power':
    train_dataset = PowerDataset()
    valid_dataset = PowerDataset()
    model = ALPACA(x_dim=1, y_dim=1, sig_eps=0.001, hidden_dim=32, final_dim=32, activation='relu')


model = model.cuda()

loss_history = []
lrs = [1e-3, 1e-4, 1e-5]
opt = torch.optim.Adam(model.parameters(), lr=lrs[0])
for lr in lrs:
    for p in opt.param_groups:
            p['lr'] = lr
    
    loss_history += train(model, opt, train_dataset, batch_size=128, support_size=5, query_size=5, train_batches=30000)
    _ = plt.plot(loss_history)
    _ = plt.yscale('log')
    _ = plt.grid()
    show_plot()

print('final_loss=%f' % np.mean(loss_history[-100:]), file=sys.stderr)

# --
# Plot example

x_s, y_s, _, _, fn = valid_dataset.sample_one(support_size=5, query_size=0)

x_grid = np.linspace(*valid_dataset.x_range, 1000)
y_grid = fn(x_grid)

inp = list2tensors((x_s, y_s, x_grid[...,None]), cuda=model.is_cuda)
mu, sig, _ = model(*inp)
mu, sig = tensors2list((mu, sig), squeeze=True)

_ = plt.scatter(x_s, y_s, c='black')
_ = plt.plot(x_grid, y_grid, c='black')
_ = plt.plot(x_grid, mu)
_ = plt.fill_between(x_grid, mu - 1.96 * np.sqrt(sig), mu + 1.96 * np.sqrt(sig), alpha=0.2)
_ = plt.xlim(*valid_dataset.x_range)
show_plot()

