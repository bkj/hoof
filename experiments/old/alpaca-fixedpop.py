#!/usr/bin/env python

"""
    alpaca.py
"""

import numpy as np
from time import time
from tqdm import tqdm, trange

import torch
from torch import nn

from rsub import *
from matplotlib import pyplot as plt

from hoof.dataset import SinusoidDataset
from hoof.models import ALPACA
from hoof.helpers import set_seeds, to_numpy

torch.set_num_threads(1)

set_seeds(123)

# --
# Helpers

def mse(act, pred):
    return float(((act - pred) ** 2).mean())


def train(model, opt, dataset, batch_size=10, train_samples=5, test_samples=5, num_batches=100):
    loss_history = []
    gen = trange(num_batches // batch_size)
    for i in gen:
        x_c, y_c, x, y, _ = dataset.sample(
            n_funcs=batch_size,
            train_samples=train_samples, # Could sample this horizon for robustness
            test_samples=test_samples,   # Could sample this horizon for robustness
        )
        
        x_c, y_c, x, y = list(map(torch.Tensor, (x_c, y_c, x, y)))
        # x_c, y_c, x, y = list(map(lambda x: x.cuda(), (x_c, y_c, x, y)))
        
        opt.zero_grad()
        mu, sig, loss = model(x_c, y_c, x, y)
        loss.backward()
        opt.step()
        
        loss_history.append(mse(mu, y))
        
        if not i % 32:
            gen.set_postfix(loss='%0.8f' % np.mean(loss_history[-100:]))
        
    return loss_history


def valid(model, dataset, batch_size=10, train_samples=5, test_samples=5, num_batches=100):
    loss_history = []
    gen = trange(num_batches // batch_size)
    for i in gen:
        x_c, y_c, x, y, _ = dataset.sample(
            n_funcs=batch_size,
            train_samples=train_samples, # Could sample this horizon for robustness
            test_samples=test_samples,   # Could sample this horizon for robustness
        )
        
        x_c, y_c, x, y = list(map(torch.Tensor, (x_c, y_c, x, y)))
        # x_c, y_c, x, y = list(map(lambda x: x.cuda(), (x_c, y_c, x, y)))
        
        with torch.no_grad():
            mu, sig, loss = model(x_c, y_c, x, y)
        
        loss_history.append(mse(mu, y))
        
        if not i % 32:
            gen.set_postfix(loss='%0.8f' % np.mean(loss_history[-100:]))
        
    return loss_history

# --
# Train

train_dataset = SinusoidDataset(sig_eps=0.0, popsize=512)
valid_dataset = SinusoidDataset(sig_eps=0.0)

model = ALPACA(x_dim=1, y_dim=1, sig_eps=0.01)

train_history, valid_history = [], []

lrs = [1e-3, 1e-4, 1e-5]
opt = torch.optim.Adam(model.parameters(), lr=lrs[0])
for lr in lrs:
    for p in opt.param_groups:
            p['lr'] = lr
    
    train_history += train(model, opt, train_dataset, batch_size=128, num_batches=30000)
    valid_history += valid(model, valid_dataset, batch_size=128, num_batches=30000)
    
    _ = plt.plot(train_history, alpha=0.75, c='red', label='train')
    _ = plt.plot(valid_history, alpha=0.75, c='green', label='valid')
    _ = plt.yscale('log')
    _ = plt.legend()
    _ = plt.grid()
    show_plot()

all_hist[popsize] = {
    "train" : train_history,
    "valid" : valid_history,
}

# --
# Plot example

dataset = valid_dataset

x_c, y_c, _, _, fns = dataset.sample(n_funcs=1, train_samples=5, test_samples=0)

x_eval = np.arange(*dataset.x_range, 0.01)
y_act  = fns[0](x_eval)

x_c, y_c, x_eval = list(map(torch.Tensor, [x_c, y_c, x_eval]))
x_c, y_c, x_eval = list(map(lambda x: x.cuda(), [x_c, y_c, x_eval]))

mu, sig, _ = model(x_c, y_c, x_eval.unsqueeze(0).unsqueeze(-1))

mu, sig, x_c, y_c, x_eval = list(map(lambda x: to_numpy(x).squeeze(), [mu, sig, x_c, y_c, x_eval]))

_ = plt.scatter(x_c, y_c, c='black')
_ = plt.plot(x_eval, y_act, c='black')
_ = plt.plot(x_eval, mu)
_ = plt.fill_between(x_eval, mu - 1.96 * np.sqrt(sig), mu + 1.96 * np.sqrt(sig), alpha=0.2)
_ = plt.xlim(*dataset.x_range)
show_plot()

