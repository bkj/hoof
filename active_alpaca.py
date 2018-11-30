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
        _ = torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        opt.step()
        
        mse = float(((mu - y) ** 2).mean())
        loss_history.append(mse)
        
        if not i % 100:
            gen.set_postfix(loss=np.median(loss_history[-100:]))
        
    return loss_history


def active_batch(model, dataset, batch_size, train_samples, test_samples):
    x_c, y_c, x, y, fns = dataset.sample(
        n_funcs=batch_size,
        train_samples=1,
        test_samples=test_samples,
    )
    
    x_c, y_c, x, y = list(map(torch.FloatTensor, (x_c, y_c, x, y)))
    
    x_grid = torch.FloatTensor(torch.arange(-5, 5, 0.1))
    x_grid_batch = x_grid.view(1, x_grid.shape[0], 1)
    x_grid_batch = x_grid_batch.repeat(batch_size, 1, 1)
    
    for nobs in range(1, train_samples):
        
        _, sig, _ = model(x_c, y_c, x_grid_batch)
        
        new_x = x_grid[sig.squeeze().max(dim=-1)[1]]
        new_y = torch.stack([f(xx) for xx, f in zip(new_x, fns)])
        
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
        
        if not i % 100:
            gen.set_postfix(loss=np.median(loss_history[-100:]))
        
    return loss_history

# --
# Params

init_steps   = 500
active_steps = 1000

train_samples = 5
test_samples  = 10

dataset = SinusoidDataset(sigma_eps=0.00)

# --
# Active learning

model = ALPACA(x_dim=1, y_dim=1, sig_eps=0.02) # fixed sig_eps is sortof cheating

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
active_loss_history = train(model, opt, dataset, train_steps=init_steps,
    test_samples=test_samples, 
    train_samples=train_samples)

opt = torch.optim.Adam(model.parameters(), lr=1e-4)
active_loss_history += active_train(model, opt, dataset, train_steps=active_steps,
    test_samples=test_samples, 
    train_samples=train_samples)

# --
# Control

model = ALPACA(x_dim=1, y_dim=1, sig_eps=0.02) # fixed sig_eps is sortof cheating

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
control_loss_history = train(model, opt, dataset, train_steps=init_steps,
    test_samples=test_samples, 
    train_samples=train_samples)

opt = torch.optim.Adam(model.parameters(), lr=1e-4)
control_loss_history += train(model, opt, dataset, train_steps=active_steps,
    test_samples=test_samples, 
    train_samples=train_samples)

# --
# Plot

_ = plt.plot(active_loss_history, alpha=0.5, label='active')
_ = plt.plot(control_loss_history, alpha=0.5, label='control')
_ = plt.axvline(init_steps, c='red')
_ = plt.yscale('log')
_ = plt.ylim(1e-4, 1e1)
_ = plt.grid()
_ = plt.legend()
show_plot()

# --
# Plot example

x_c, y_c, _, _, fns = active_batch(
    model=model,
    dataset=dataset,
    batch_size=10, 
    train_samples=train_samples,
    test_samples=test_samples
)

x_c, y_c = x_c[:1], y_c[:1]

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

for i, txt in enumerate(range(train_samples)):
    _ = plt.annotate(txt, (float(x_c[i]), float(y_c[i])), color='red')

show_plot()
