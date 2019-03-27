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

from hoof.dataset import SinusoidDataset, AbsValDataset, PowerDataset, PQuadDataset
from hoof.models import ALPACA
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

# dataset = SinusoidDataset(sig_eps=0.00)
# dataset = AbsValDataset(popsize=100)
# dataset = PowerDataset()
train_dataset = PQuadDataset(x_range=[-5, 5], popsize=30)
model         = ALPACA(x_dim=1, y_dim=1, sig_eps=0.2, hidden_dim=32, final_dim=32, activation='tanh')

model     = model.cuda()
model.eye = model.eye.cuda()

loss_history = []
lrs = [1e-3, 1e-3, 1e-4, 1e-5]
opt = torch.optim.Adam(model.parameters(), lr=lrs[0])
for lr in lrs:
    for p in opt.param_groups:
            p['lr'] = lr
    
    loss_history += train(model, opt, train_dataset, train_samples=5, test_samples=10, batch_size=128, train_batches=30000)
    _ = plt.plot(loss_history)
    _ = plt.yscale('log')
    _ = plt.grid()
    show_plot()




# --
# Plot example

dataset = PQuadDataset(x_range=[-5, 5])
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

# --
# Optimization

from scipy.stats import norm
def gaussian_ei(mu, sig, y_opt=0.0):
    """ gaussian_ei for minimizing a function """
    values       = np.zeros_like(mu)
    mask         = sig > 0
    improve      = y_opt - mu[mask]
    Z            = improve / sig[mask]
    exploit      = improve * norm.cdf(Z)
    explore      = sig[mask] * norm.pdf(Z)
    values[mask] = exploit + explore
    return values


all_hist = []
for _ in trange(100):
    x_c, y_c, _, _, fns = dataset.sample(n_funcs=1, train_samples=5, test_samples=0)
    
    x_eval = np.arange(*dataset.x_range, 0.0001)
    y_act  = fns[0](x_eval)
    
    x_eval = np.hstack([x_eval, x_c.squeeze()])
    y_act  = np.hstack([y_act, y_c.squeeze()])
    
    hist = [y_c.min() - y_act.min()]
    for _ in range(5):
        x_c_, y_c_, x_eval_ = list(map(lambda x: torch.Tensor(x).cuda(), [x_c, y_c, x_eval]))
        
        mu_, sig_, _ = model(x_c_, y_c_, x_eval_.unsqueeze(0).unsqueeze(-1))
        
        mu, sig = list(map(lambda x: to_numpy(x).squeeze(), [mu_, sig_]))
        
        ei = gaussian_ei(mu, sig, y_opt=y_c.min())
        # if ei.max() == 0:
        #     break
        
        next_x = x_eval[np.argmax(ei)]
        next_y = y_act[np.argmax(ei)]
        
        x_c = np.hstack([x_c, [[[next_x]]]])
        y_c = np.hstack([y_c, [[[next_y]]]])
        
        # _ = plt.scatter(x_c, y_c, c='black', alpha=0.25)
        # _ = plt.plot(x_eval, y_act, c='black')
        # _ = plt.plot(x_eval, mu)
        # _ = plt.fill_between(x_eval, mu - 1.96 * np.sqrt(sig), mu + 1.96 * np.sqrt(sig), alpha=0.2)
        # _ = plt.xlim(*dataset.x_range)
        
        # y_act  = y_act[x_eval != next_x]
        # x_eval = x_eval[x_eval != next_x]
        assert y_c.min() >= y_act.min()
        hist.append(y_c.min() - y_act.min())
    
    # _ = plt.plot(hist)
    all_hist.append(hist[-1])

all_hist = np.array(all_hist)
(all_hist == 0).mean()
np.log10(all_hist[all_hist != 0]).mean()


all_hist = []
for _ in trange(100):
    x_c, y_c, _, _, fns = dataset.sample(n_funcs=1, train_samples=10, test_samples=0)
    
    x_eval = np.arange(*dataset.x_range, 0.0001)
    y_act  = fns[0](x_eval)
    
    x_eval = np.hstack([x_eval, x_c.squeeze()])
    y_act  = np.hstack([y_act, y_c.squeeze()])
    
    hist = [y_c.min() - y_act.min()]
    all_hist.append(hist[-1])

all_hist = np.array(all_hist)
(all_hist == 0).mean()

np.log10(all_hist).mean()










