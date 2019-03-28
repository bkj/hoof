#!/usr/bin/env python

"""
    simple_example.py
"""

import sys
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm, trange

import torch
from torch import nn

from rsub import *
from matplotlib import pyplot as plt

from hoof import dataset
from hoof.models import ALPACA
from hoof.helpers import set_seeds, to_numpy, list2tensors, tensors2list, set_lr
from hoof.bayesopt import gaussian_ei, scipy_minimize

torch.set_num_threads(1)
set_seeds(345)

# --
# Dataset

dataset_name = 'QuadraticDataset'
popsize = 30

dataset_cls   = getattr(dataset, dataset_name)
train_dataset = dataset_cls(popsize=popsize, x_dim=3, x_range=[-3, 3])
valid_dataset = dataset_cls(x_dim=3, x_range=[-3, 3])

# >>
# Plot draws from dataset

# for _ in range(100):
#     x, y, *_ = valid_dataset.sample_one(support_size=100, query_size=0)
#     x, y = x.squeeze(), y.squeeze()
#     _ = plt.plot(x[np.argsort(x)], y[np.argsort(x)], alpha=0.25)

# show_plot()

# <<

# --
# Train

model = ALPACA(input_dim=train_dataset.x_dim, output_dim=1, sig_eps=0.1, hidden_dim=128, activation='relu').cuda()

train_history = []
lrs = [1e-4, 1e-4, 1e-5]
opt = torch.optim.Adam(model.parameters(), lr=lrs[0])

train_kwargs = {"batch_size" : 30, "support_size" : 10, "query_size" : 1000, "num_samples" : 30000}

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

# --
# Plot example

if valid_dataset.x_dim == 1:
    x_s, y_s, _, _, fn = valid_dataset.sample_one(support_size=train_kwargs['support_size'], query_size=0)
    x_grid = np.linspace(*valid_dataset.x_range, 1000).reshape(-1, 1)
    y_grid = fn(x_grid)
    
    inp = list2tensors((x_s, y_s, x_grid), cuda=model.is_cuda)
    mu, sig, _ = model(*inp)
    mu, sig = tensors2list((mu, sig), squeeze=True)
    
    _ = plt.scatter(x_s, y_s, c='black')
    _ = plt.plot(x_grid, y_grid, c='black')
    _ = plt.plot(x_grid, mu)
    _ = plt.fill_between(x_grid.squeeze(), mu - 1.96 * np.sqrt(sig), mu + 1.96 * np.sqrt(sig), alpha=0.2)
    _ = plt.xlim(*valid_dataset.x_range)
    show_plot()

# --
# Run BO experiment

def do_bayesopt(model, fn, dataset, num_samples=20, num_burnin=10, num_candidates_per_round=100000):
    # !! What's the best way to maximize the surrogate function?
    
    x = dataset.sample_x(n=num_burnin)
    y = fn(x)
    
    traj = []
    for _ in range(num_samples - num_burnin):
        x_cand = dataset.sample_x(n=num_candidates_per_round)
        
        inp = list2tensors((x, y, x_cand), cuda=model.is_cuda)
        mu, sig, _ = model(*inp)
        mu, sig = tensors2list((mu, sig), squeeze=True)
        
        eis = gaussian_ei(mu, sig, incumbent=y.min())
        
        best_x = x_cand[eis.argmax(), None]
        
        x = np.vstack([x, best_x])
        y = np.vstack([y, fn(best_x)])
    
    return x, y

def run_logmin(x):
    return np.log10(pd.Series(x).cummin().values)


bo_hist, rand_hist = [], []
for _ in trange(32):
    _, _, _, _, fn = valid_dataset.sample_one(support_size=0, query_size=0)
    x_bo, y_bo = do_bayesopt(model, fn, valid_dataset, num_burnin=10, num_samples=50)
    
    x_rand = valid_dataset.sample_x(n=100)
    y_rand = fn(x_rand)
    
    x_opt, y_opt = scipy_minimize(fn, x_s[0, None])
    
    bo_hist.append(run_logmin(y_bo.squeeze() - y_opt))
    rand_hist.append(run_logmin(y_rand.squeeze() - y_opt))


for bo in bo_hist:
    _ = plt.plot(bo, c='green', alpha=0.05)

_ = plt.plot(np.stack(bo_hist).mean(axis=0), c='green')

for r in rand_hist:
    _ = plt.plot(r, c='red', alpha=0.05)

_ = plt.plot(np.stack(rand_hist).mean(axis=0), c='red')

show_plot()


