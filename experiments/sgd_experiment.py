#!/usr/bin/env python

"""
    sgd_experiment.py
"""

import numpy as np
from tqdm import tqdm

import torch
from torch import nn

from rsub import *
from matplotlib import pyplot as plt

from dataset import CacheDataset, SGDDataset
from models import ALPACA
from helpers import set_seeds, to_numpy

torch.set_num_threads(1)

set_seeds(123)

XMIN = -5
XMAX = 0

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
        
        if not i % 10:
            gen.set_postfix(loss=np.mean(loss_history[-10:]))
        
    return loss_history


def active_batch(model, dataset, batch_size, train_samples, test_samples):
    x_c, y_c, x, y, fns = dataset.sample(
        n_funcs=batch_size,
        train_samples=1,
        test_samples=test_samples,
    )
    
    x_c, y_c, x, y = list(map(torch.FloatTensor, (x_c, y_c, x, y)))
    
    x_grid = torch.FloatTensor(torch.arange(XMIN, XMAX, 0.1))
    x_grid_batch = x_grid.view(1, x_grid.shape[0], 1)
    x_grid_batch = x_grid_batch.repeat(batch_size, 1, 1)
    
    for nobs in range(1, train_samples):
        
        mu, sig, _ = model(x_c, y_c, x_grid_batch)
        
        # _ = plt.plot(to_numpy(x_grid), to_numpy(mu[0].squeeze() + 1.96 * sig[0].squeeze()))
        # _ = plt.scatter(to_numpy(x_c[0]), to_numpy(y_c[0]))
        # show_plot()
        
        sel = sig.squeeze().max(dim=-1)[1]
        # sel = (mu.squeeze() + 1.96 * sig.squeeze()).max(dim=-1)[1] # maximum upper bound
        
        new_x = x_grid[sel]
        # >>
        new_x = new_x.view(-1, 1)
        new_y = np.hstack([f(to_numpy(xx)) for xx, f in zip(new_x, fns)])
        new_y = torch.FloatTensor(new_y)
        # <<
        
        x_c = torch.cat([x_c, new_x.view(batch_size, 1, 1)], dim=1)
        y_c = torch.cat([y_c, new_y.view(batch_size, 1, 1)], dim=1)
    
    return x_c, y_c, x, y, fns


def active_train(model, opt, dataset, batch_size=10, train_samples=5, test_samples=5, train_steps=2000):
    loss_history = []
    gen = tqdm(range(train_steps))
    for i in gen:
        x_c, y_c, x, y, _ = active_batch(
            model=model,
            dataset=dataset,
            batch_size=batch_size,
            train_samples=train_samples, # Can sample these for robustness
            test_samples=test_samples,   # Can sample these for robustness
        )
        
        opt.zero_grad()
        mu, sig, loss = model(x_c, y_c, x, y)
        loss.backward()
        _ = torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        opt.step()
        
        mse = float(((mu - y) ** 2).mean())
        loss_history.append(mse)
        
        if not i % 10:
            gen.set_postfix(loss=np.median(loss_history[-10:]))
        
    return loss_history


# --
# Train

steps1 = 1000
steps2 = 100

test_samples  = 5
train_samples = 5

dataset       = SGDDataset(x_range=[XMIN, XMAX])
cache_dataset = CacheDataset(dataset, n_batches=steps1, n_funcs=10, train_samples=5, test_samples=20)

model = ALPACA(x_dim=1, y_dim=1, sig_eps=1e-5, activation='relu') # fixed sig_eps is sortof cheating

opt = torch.optim.Adam(model.parameters(), lr=1e-5)
loss_history = train(
    model=model,
    opt=opt,
    dataset=cache_dataset,
    train_steps=steps1
)

opt = torch.optim.Adam(model.parameters(), lr=1e-5)
loss_history += active_train(
    model=model,
    opt=opt,
    dataset=dataset,
    train_steps=steps2,
    test_samples=test_samples, 
    train_samples=train_samples
)

_ = plt.plot(loss_history)
_ = plt.yscale('log')
# _ = plt.ylim(1e-4, 1e1)
_ = plt.grid()
show_plot()


# --
# Plot example

x_c, y_c, _, _, fns = dataset.sample(n_funcs=1, train_samples=5, test_samples=1)

x_eval = np.arange(XMIN, XMAX, 0.1)
y_act  = fns[0](x_eval)

x_c, y_c, x_eval = list(map(torch.FloatTensor, [x_c, y_c, x_eval]))
mu, sig, _ = model(x_c, y_c, x_eval.unsqueeze(0).unsqueeze(-1))

mu, sig, x_c, y_c, x_eval = list(map(lambda x: to_numpy(x).squeeze(), [mu, sig, x_c, y_c, x_eval]))

_ = plt.scatter(x_c, y_c, c='black')
_ = plt.plot(x_eval, y_act, c='black')
_ = plt.plot(x_eval, mu)
_ = plt.fill_between(x_eval, mu - 1.96 * np.sqrt(sig), mu + 1.96 * np.sqrt(sig), alpha=0.2)
_ = plt.xlim(x_eval.min(), x_eval.max())
# _ = plt.ylim(0, 1)

show_plot()

# --
# Plot active example

x_c, y_c, _, _, fns = active_batch(
    model=model,
    dataset=dataset,
    batch_size=10, 
    train_samples=train_samples,
    test_samples=test_samples
)

x_c, y_c = x_c[:1], y_c[:1]

x_eval = np.arange(XMIN, XMAX, 0.1)
y_act  = fns[0](x_eval)

x_c, y_c, x_eval = list(map(torch.FloatTensor, [x_c, y_c, x_eval]))
mu, sig, _ = model(x_c, y_c, x_eval.unsqueeze(0).unsqueeze(-1))

mu, sig, x_c, y_c, x_eval = list(map(lambda x: to_numpy(x).squeeze(), [mu, sig, x_c, y_c, x_eval]))

_ = plt.scatter(x_c, y_c, c='black')
_ = plt.plot(x_eval, y_act, c='black')
_ = plt.plot(x_eval, mu)
_ = plt.fill_between(x_eval, mu - 1.96 * np.sqrt(sig), mu + 1.96 * np.sqrt(sig), alpha=0.2)
_ = plt.xlim(XMIN, XMAX)
# _ = plt.ylim(0, 1)

for i, txt in enumerate(range(train_samples)):
    _ = plt.annotate(txt, (float(x_c[i]), float(y_c[i])), color='red')

show_plot()
