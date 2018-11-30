#!/usr/bin/env python

"""
    alpaca.py
"""

import numpy as np
from tqdm import tqdm

import torch
from torch import nn

from rsub import *
from matplotlib import pyplot as plt

from dataset import SinusoidDataset
from models import ALPACA

torch.set_num_threads(1)

_ = np.random.seed(123)
_ = torch.manual_seed(123 + 111)
_ = torch.cuda.manual_seed(123 + 222)

def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    else:
        return x.cpu().detach().numpy()

# --
# Helpers

def train(model, opt, dataset, batch_size=10, train_samples=5, test_samples=5, train_steps=2000):
    loss_history = []
    gen = tqdm(range(train_steps))
    for i in gen:
        x_c, y_c, x, y, _ = dataset.sample(
            n_funcs=batch_size,
            train_samples=train_samples, # Can sample these for robustness
            test_samples=test_samples,   # Can sample these for robustness
        )
        
        x_c, y_c, x, y = list(map(torch.FloatTensor, (x_c, y_c, x, y)))
        
        opt.zero_grad()
        mu, sig, loss = model(x_c, y_c, x, y)
        loss.backward()
        opt.step()
        
        mse = float(((mu - y) ** 2).mean())
        loss_history.append(mse)
        
        if not i % 100:
            gen.set_postfix(loss=np.mean(loss_history[-100:]))
        
    return loss_history

# --
# Train

dataset = SinusoidDataset(sigma_eps=0.00)
model   = ALPACA(x_dim=1, y_dim=1, sig_eps=0.02) # fixed sig_eps is sortof cheating
opt     = torch.optim.Adam(model.parameters(), lr=1e-3)

loss_history = train(model, opt, dataset)

_ = plt.plot(loss_history)
_ = plt.yscale('log')
_ = plt.ylim(1e-4, 1e1)
_ = plt.grid()
show_plot()

np.mean(loss_history[-200:])

# --
# Plot example

x_c, y_c, _, _, fns = dataset.sample(n_funcs=1, train_samples=5, test_samples=0)

x_eval = np.arange(-5, 5, 0.1)
y_act  = fns[0](x_eval, noise=False)

x_c, y_c, x_eval = list(map(torch.FloatTensor, [x_c, y_c, x_eval]))
mu, sig, _ = model(x_c, y_c, x_eval.unsqueeze(0).unsqueeze(-1))

mu, sig, x_c, y_c, x_eval = list(map(lambda x: to_numpy(x).squeeze(), [mu, sig, x_c, y_c, x_eval]))

_ = plt.scatter(x_c, y_c, c='black')
_ = plt.plot(x_eval, y_act, c='black')
_ = plt.plot(x_eval, mu)
_ = plt.fill_between(x_eval, mu - 1.96 * np.sqrt(sig), mu + 1.96 * np.sqrt(sig), alpha=0.2)
_ = plt.xlim(-5, 5)
show_plot()

