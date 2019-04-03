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
set_seeds(111)

# --
# Dataset

dataset_name = 'SinusoidDataset'
popsize = None

dataset_cls = getattr(dataset, dataset_name)

train_dataset = dataset_cls(popsize=popsize)
valid_dataset = dataset_cls()

# Plot some examples
# x_s, y_s, _, _, _ = valid_dataset.sample_one(support_size=100, query_size=0)
# _ = plt.scatter(x_s.squeeze(), y_s.squeeze())
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

model = ALPACA(input_dim=1, output_dim=1, sig_eps=1, hidden_dim=128, activation='tanh').cuda()

train_history = []
lrs = [1e-4, 1e-5]
opt = torch.optim.Adam(model.parameters(), lr=lrs[0])

train_kwargs = {"batch_size" : 64, "support_size" : 10, "query_size" : 100, "num_samples" : 30000}

for lr in lrs:
    set_lr(opt, lr)
    
    train_history += model.do_train(dataset=train_dataset, opt=opt, **train_kwargs)
    
    _ = plt.plot(train_history, c='red', label='train')
    _ = plt.yscale('log')
    _ = plt.grid()
    _ = plt.legend()
    show_plot()


valid_history = model.do_valid(dataset=valid_dataset, **train_kwargs)

print('final_train_loss=%f' % np.mean(train_history[-100:]), file=sys.stderr)
print('final_valid_loss=%f' % np.mean(valid_history[-100:]), file=sys.stderr)

# >>

from scipy.optimize import minimize

x_s, y_s, x_q, y_q, fn = valid_dataset.sample_one(support_size=20, query_size=20)
y_s += np.random.normal(0, 1, y_s.shape) * 0.1
y_q += np.random.normal(0, 1, y_q.shape) * 0.1

model = model.eval()
model = model.cpu()

inp = list2tensors((x_s, y_s, x_q, y_q), cuda=model.is_cuda)
init = float(model(*inp)[-1])
print(init)

inp = list2tensors((x_s, y_s, x_s, y_s), cuda=model.is_cuda)

model = model.eval()
model = model.cpu()

def _target(p):
    model.blr.sig_eps     = float(10 ** p)
    model.blr.log_sig_eps = np.log(float(10 ** p))
    return float(model(*inp)[-1])



opt_res = minimize(_target, np.array([-1]), method='Powell', bounds=[(-10, 2)])

sig_eps = float(10 ** opt_res.x)

sig_eps

model.blr.sig_eps     = sig_eps
model.blr.log_sig_eps = np.log(sig_eps)
float(model(*inp)[-1])

inp = list2tensors((x_s, y_s, x_q, y_q), cuda=model.is_cuda)
float(model(*inp)[-1])



# <<

# --
# Plot example

# inp_target = list2tensors((x_s, y_s, x_s, y_s), cuda=model.is_cuda)
# _target    = make_target(model, inp_target)
# _ = plt.plot([_target(np.array([i])) for i in np.linspace(-3, 0)])
# show_plot()

np.random.seed(456)
valid_dataset.set_seed(123)
fig, ax = plt.subplots(3, 3)

def make_target(model, inp):
    def _target(p):
        model.blr.sig_eps     = float(10 ** p)
        model.blr.log_sig_eps = np.log(float(10 ** p))
        return float(model(*inp)[-1])
    
    return _target


for i in range(3):
    for j in range(3):
        x_s, y_s, _, _, fn = valid_dataset.sample_one(support_size=20, query_size=0)
        x_grid = np.linspace(*valid_dataset.x_range, 1000).reshape(-1, 1)
        y_grid = fn(x_grid)
        
        y_s += np.random.normal(0, 1, y_s.shape) * 0.1
        
        # >>
        # Tune sig_eps
        
        model.blr.sig_eps     = 1
        model.blr.log_sig_eps = 0
        
        tune_sig_eps = True
        if tune_sig_eps:
            inp_target = list2tensors((x_s, y_s, x_s, y_s), cuda=model.is_cuda)
            _target    = make_target(model, inp_target)
            opt_res    = minimize(_target, np.array([-1]), method='Powell', bounds=[(-10, 2)])
            sig_eps    = float(10 ** opt_res.x)
            model.blr.sig_eps     = sig_eps
            model.blr.log_sig_eps = np.log(sig_eps)
        
        # <<
        
        inp = list2tensors((x_s, y_s, x_grid, y_grid), cuda=model.is_cuda)
        mu, sig, nll = model(*inp)
        print(float(nll))
        mu, sig = tensors2list((mu, sig), squeeze=True)
        
        _ = ax[i,j].plot(x_grid, y_grid, c='black', alpha=0.25)
        _ = ax[i,j].plot(x_grid, mu)
        # _ = ax[i,j].plot(x_grid, y_prior.squeeze())
        # _ = ax[i,j].plot(x_grid, rks_regression(x_s, y_s, x_grid, n_components=50, gamma=0.25), c='orange', alpha=0.75)
        _ = ax[i,j].fill_between(x_grid.squeeze(), mu - 1.96 * np.sqrt(sig), mu + 1.96 * np.sqrt(sig), alpha=0.2)
        _ = ax[i,j].scatter(x_s, y_s, c='red')
        _ = ax[i,j].set_xlim(*valid_dataset.x_range)

show_plot()

