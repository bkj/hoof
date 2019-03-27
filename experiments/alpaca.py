#!/usr/bin/env python

"""
    alpaca.py
"""

import numpy as np
from time import time
from tqdm import tqdm, trange

import torch
from torch import nn
# torch.set_default_tensor_type('torch.DoubleTensor')

from rsub import *
from matplotlib import pyplot as plt

from hoof.dataset import SinusoidDataset, PowerDataset, QuadraticDataset
from hoof.dataset import CacheDataset
from hoof.models import ALPACA, ALPACA2
from hoof.helpers import set_seeds, to_numpy

torch.set_num_threads(1)

set_seeds(123)

# --
# Helpers

def mse(act, pred):
    return float(((act - pred) ** 2).mean())

def train(model, opt, dataset, batch_size=10, train_samples=5, test_samples=5, train_batches=100):
    loss_history = []
    gen = trange(train_batches // batch_size)
    for i in gen:
        x_c, y_c, x, y, _ = dataset.sample(
            n_funcs=batch_size,
            train_samples=train_samples, # Could sample this horizon for robustness
            test_samples=test_samples,   # Could sample this horizon for robustness
        )
        
        x_c, y_c, x, y = list(map(torch.Tensor, (x_c, y_c, x, y)))
        x_c, y_c, x, y = list(map(lambda x: x.cuda(), (x_c, y_c, x, y)))
        
        opt.zero_grad()
        mu, sig, loss = model(x_c, y_c, x, y)
        loss.backward()
        opt.step()
        
        loss_history.append(mse(mu, y))
        
        if not i % 32:
            gen.set_postfix(loss='%0.8f' % np.mean(loss_history[-100:]))
        
    return loss_history



# --
# Train

dataset = SinusoidDataset(sig_eps=0.00)
model   = ALPACA(x_dim=1, y_dim=1, sig_eps=0.01) # fixed sig_eps is sortof cheating

model     = model.cuda()
model.eye = model.eye.cuda()

loss_history = []
for lr in [1e-3, 1e-4, 1e-5]:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history += train(model, opt, dataset, batch_size=128, train_batches=30000)

_ = plt.plot(loss_history)
_ = plt.yscale('log')
_ = plt.grid()
show_plot()

# --
# Plot example

dataset = SinusoidDataset(sig_eps=0.00)
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

