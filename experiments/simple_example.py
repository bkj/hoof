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

from hoof.dataset import SinusoidDataset, PowerDataset, QuadraticDataset
from hoof.models import ALPACA
from hoof.helpers import set_seeds, to_numpy, list2tensors, tensors2list

torch.set_num_threads(1)
set_seeds(345)

# --
# Helpers

def mse(act, pred):
    return float(((act - pred) ** 2).mean())


def train(model, opt, dataset, batch_size=10, support_size=5, query_size=5, num_samples=100):
    hist = []
    gen = trange(num_samples // batch_size)
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
        
        hist.append(mse(mu, y_query))
        
        if not batch_idx % 10:
            gen.set_postfix(loss='%0.8f' % np.mean(hist[-10:]))
        
    return hist


def valid(model, dataset, batch_size=10, support_size=5, query_size=5, num_samples=100):
    hist = []
    gen = trange(num_samples // batch_size)
    for batch_idx in gen:
        x_support, y_support, x_query, y_query, _ = dataset.sample_batch(
            batch_size=batch_size,
            support_size=support_size, # Could sample this horizon for robustness
            query_size=query_size,     # Could sample this horizon for robustness
        )
        
        (x_support, y_support, x_query, y_query) = \
            list2tensors((x_support, y_support, x_query, y_query), cuda=model.is_cuda)
        
        with torch.no_grad():
            mu, sig, loss = model(x_support, y_support, x_query, y_query)
        
        hist.append(mse(mu, y_query))
        if not batch_idx % 10:
            gen.set_postfix(loss='%0.8f' % np.mean(hist[-10:]))
    
    return hist

# --
# Dataset

dataset = 'sinusoid'
popsize = None

datasets = {
    "sinusoid"  : SinusoidDataset,
    "power"     : PowerDataset,
    "quadratic" : QuadraticDataset,
}

dataset_cls = datasets[dataset]

train_dataset = dataset_cls(popsize=popsize)
valid_dataset = dataset_cls()

# --
# Train

model = ALPACA(input_dim=1, output_dim=1, sig_eps=0.01, hidden_dim=128, activation='tanh').cuda()

train_history = []
lrs = [1e-4, 1e-5]
opt = torch.optim.Adam(model.parameters(), lr=lrs[0])

train_kwargs = {"batch_size" : 128, "support_size" : 10, "query_size" : 10, "num_samples" : 30000}

for lr in lrs:
    for p in opt.param_groups:
            p['lr'] = lr
    
    train_history  += train(model, opt, train_dataset, **train_kwargs)
    
    _ = plt.plot(train_history, c='red', label='train')
    _ = plt.yscale('log')
    _ = plt.grid()
    _ = plt.legend()
    show_plot()


valid_history = valid(model, valid_dataset, **train_kwargs)

print('final_train_loss=%f' % np.mean(train_history[-100:]), file=sys.stderr)
print('final_valid_loss=%f' % np.mean(valid_history[-100:]), file=sys.stderr)

# --
# Plot example

# umodel = ALPACA(input_dim=1, output_dim=1, sig_eps=0.01, hidden_dim=128, activation='tanh').cuda()
# uopt = torch.optim.Adam(umodel.parameters(), lr=lrs[0])
# train(umodel, uopt, train_dataset, batch_size=128, support_size=10, query_size=10, num_samples=128000)

def rbf_pred(x_s, y_s, x_grid, **kwargs):
    rbf = RBFSampler(**kwargs)
    p_s = rbf.fit_transform(x_s)
    lr  = LinearRegression().fit(p_s, y_s)
    return lr.predict(rbf.transform(x_grid))


x_s, y_s, _, _, fn = valid_dataset.sample_one(support_size=5, query_size=0)
x_grid = np.linspace(*valid_dataset.x_range, 1000).reshape(-1, 1)
y_grid = fn(x_grid)

inp = list2tensors((x_s, y_s, x_grid), cuda=model.is_cuda)
mu, sig, _ = model(*inp)
mu, sig = tensors2list((mu, sig), squeeze=True)

# umu, usig, _ = umodel(*inp)
# umu, usig = tensors2list((umu, usig), squeeze=True)

_ = plt.scatter(x_s, y_s, c='black')
_ = plt.plot(x_grid, y_grid, c='black')
_ = plt.plot(x_grid, mu, alpha=0.75)
# _ = plt.plot(x_grid, umu, alpha=0.75)
# _ = plt.plot(x_grid, rbf_pred(x_s, y_s, x_grid, n_components=50, gamma=0.25), c='orange', alpha=0.75)
# _ = plt.plot(x_grid, rbf_pred(x_s, y_s, x_grid, n_components=50, gamma=0.5), c='orange', alpha=0.75)
# _ = plt.plot(x_grid, rbf_pred(x_s, y_s, x_grid, n_components=50, gamma=1.0), c='orange', alpha=0.75)
_ = plt.fill_between(x_grid.squeeze(), mu - 1.96 * np.sqrt(sig), mu + 1.96 * np.sqrt(sig), alpha=0.2)
# _ = plt.fill_between(x_grid.squeeze(), umu - 1.96 * np.sqrt(sig), umu + 1.96 * np.sqrt(sig), alpha=0.2)
_ = plt.xlim(*valid_dataset.x_range)
show_plot()

# >>
# RKS implementation
from hoof.models import BLR
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LinearRegression

x_s, y_s, _, _, fn = valid_dataset.sample_one(support_size=10, query_size=0)
x_grid = np.linspace(*valid_dataset.x_range, 1000).reshape(-1, 1)
y_grid = fn(x_grid)




_ = plt.scatter(x_s, y_s, c='black')
_ = plt.plot(x_grid, y_grid, c='black', alpha=0.75)
_ = plt.plot(x_grid, y_pred + np.random.uniform(0, 0.1, y_grid.shape), c='red', alpha=0.75)
show_plot()


# <<




