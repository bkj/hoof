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


def active_batch(model, dataset, batch_size=10, train_samples=5, test_samples=5):
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
# Init + active learning

init_steps   = 500
active_steps = 500

dataset = SinusoidDataset(sigma_eps=0.00)
model   = ALPACA(x_dim=1, y_dim=1, sig_eps=0.02) # fixed sig_eps is sortof cheating

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
active_loss_history = train(model, opt, dataset, train_steps=init_steps)

opt = torch.optim.Adam(model.parameters(), lr=1e-4)
active_loss_history += active_train(model, opt, dataset, train_steps=active_steps)

# _ = plt.plot(active_loss_history)
# _ = plt.axvline(init_steps, c='red')
# _ = plt.yscale('log')
# _ = plt.ylim(1e-4, 1e1)
# _ = plt.grid()
# show_plot()

# --
# Init + init (control)

init_steps   = 500
active_steps = 500

dataset = SinusoidDataset(sigma_eps=0.00)
model   = ALPACA(x_dim=1, y_dim=1, sig_eps=0.02) # fixed sig_eps is sortof cheating

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
control_loss_history = train(model, opt, dataset, train_steps=init_steps)

opt = torch.optim.Adam(model.parameters(), lr=1e-4)
# active_loss_history = active_train(model, opt, dataset, train_steps=active_steps)
control_loss_history += train(model, opt, dataset, train_steps=active_steps) # !!

# _ = plt.plot(control_loss_history)
# _ = plt.axvline(init_steps, c='red')
# _ = plt.yscale('log')
# _ = plt.ylim(1e-4, 1e1)
# _ = plt.grid()
# show_plot()

# --

_ = plt.plot(active_loss_history, alpha=0.5, label='active')
_ = plt.plot(control_loss_history, alpha=0.5, label='control')
_ = plt.axvline(init_steps, c='red')
_ = plt.yscale('log')
_ = plt.ylim(1e-4, 1e1)
_ = plt.grid()
_ = plt.legend()
show_plot()








# # --
# # Active




# horizon = 5
# test_horizon = 5
# batch_size = 10
# num_train_updates = 3000
# opt = torch.optim.Adam(model.parameters(), lr=1e-4)

# x_grid = torch.FloatTensor(torch.arange(-5, 5, 0.1))
# x_grid = x_grid.view(1, x_grid.shape[0], 1)

# gen = tqdm(range(num_train_updates))
# for i in gen:
    
#     generating_functions = [random_sine() for _ in range(batch_size)]
#     x_c = np.random.uniform(-5, 5, batch_size)
#     y_c = [f(xx) for xx, f in zip(x_c, generating_functions)]
    
#     x_c = torch.FloatTensor(x_c).view(batch_size, 1, 1)
#     y_c = torch.FloatTensor(y_c).view(batch_size, 1, 1)
    
#     for nobs in range(1, horizon):
        
#         mu, sig, _ = model(x_c, y_c, x_grid.repeat(batch_size, 1, 1))
        
#         new_x = x_grid[0][sig.squeeze().max(dim=-1)[1]]
#         new_y = torch.cat([f(xx) for xx, f in zip(new_x, generating_functions)])
        
#         x_c = torch.cat([x_c, new_x.view(batch_size, 1, 1)], dim=1)
#         y_c = torch.cat([y_c, new_y.view(batch_size, 1, 1)], dim=1)
    
#     x = [np.random.uniform(-5, 5, test_horizon) for f in generating_functions]
#     y = [f(xx) for xx, f in zip(x, generating_functions)]
    
#     x = torch.FloatTensor(np.vstack(x).reshape(batch_size, test_horizon, 1))
#     y = torch.FloatTensor(np.vstack(y).reshape(batch_size, test_horizon, 1))
    
#     opt.zero_grad()
#     mu, sig, loss = model(x_c, y_c, x, y)
#     loss.backward()
#     _ = torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
#     opt.step()
    
#     mse = float(((mu - y) ** 2).mean())
#     loss_history.append(mse)
    
#     if not i % 100:
#         gen.set_postfix(loss=np.median(loss_history[-100:]))


# _ = plt.plot(loss_history[30:])
# _ = plt.yscale('log')
# _ = plt.ylim(1e-4, 1e1)
# _ = plt.grid()
# show_plot()

# print(np.mean(loss_history[-200:]))

# # # --
# # # Testing

# # generating_function = random_sine()
# # x_c = np.random.uniform(-5, 5, 1)
# # y_c = generating_function(x_c)

# # for nobs in range(1, horizon):
# #     x_c = torch.FloatTensor(x_c).view(1, nobs, 1)
# #     y_c = torch.FloatTensor(y_c).view(1, nobs, 1)
    
# #     mu, sig, _ = model(x_c, y_c, x_grid)
    
# #     # --
# #     # Plot
    
# #     mu_  = mu.squeeze().detach().numpy()
# #     sig_ = sig.squeeze().detach().numpy()
    
# #     x_eval = x_grid[0].squeeze().numpy()
# #     y_act = generating_function(x_eval)
    
# #     _ = plt.scatter(
# #         x_c.squeeze().numpy(),
# #         y_c.squeeze().numpy(),
# #         c='grey',
# #     )
# #     for i, txt in enumerate(range(nobs)):
# #         _ = plt.annotate(txt, (float(x_c[0][i]), float(y_c[0][i])))
    
# #     _ = plt.plot(x_eval, y_act, c='black')
# #     # _ = plt.plot(x_eval, 10 * sig_, c='black')
# #     _ = plt.plot(x_eval, mu_)
# #     _ = plt.fill_between(x_eval, mu_ - 1.96 * np.sqrt(sig_), mu_ + 1.96 * np.sqrt(sig_), alpha=0.2)
# #     _ = plt.xlim(-5, 5)
# #     show_plot()
    
# #     # --
# #     # Choose next point
    
# #     new_x = x_grid[0][sig.squeeze().max(dim=-1)[1]]
# #     new_y = generating_function(new_x)
    
# #     x_c = torch.cat([x_c, new_x.view(1, 1, 1)], dim=1)
# #     y_c = torch.cat([y_c, new_y.view(1, 1, 1)], dim=1)
    
# #     mu_t, sig_t, _ = model(x_c, y_c, x_grid)


