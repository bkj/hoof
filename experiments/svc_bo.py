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

from hoof.dataset import FileDataset
from hoof.models import ALPACA
from hoof.helpers import set_seeds, to_numpy, list2tensors, tensors2list

torch.set_num_threads(1)
set_seeds(345)

# --
# Helpers

def mse(act, pred):
    return float(((act - pred) ** 2).mean())

# --
# Dataset

path = 'data/topk.jl'
train_dataset = FileDataset(path=path)
valid_dataset = FileDataset(path=path)

# Non-overlapping tasks
num_tasks = len(train_dataset.task_ids)
task_ids  = np.random.permutation(train_dataset.task_ids)
train_dataset.task_ids, valid_dataset.task_ids = task_ids[:20], task_ids[20:]

# --
# Train

model = ALPACA(x_dim=train_dataset.x_dim, y_dim=1, sig_eps=0.01, hidden_dim=128, final_dim=128, activation='relu')
model = model.cuda()

train_history = []
lrs = [1e-4, 1e-5]
opt = torch.optim.Adam(model.parameters(), lr=lrs[0])

train_kwargs = {"batch_size" : 32, "support_size" : 10, "query_size" : 10, "num_batches" : 30000}

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
# BO helpers

import pandas as pd
from scipy.stats import norm
# from scipy.optimize import minimize

def gaussian_ei(mu, sig, incumbent=0.0):
    """ gaussian_ei for minimizing a function """
    values       = np.zeros_like(mu)
    mask         = sig > 0
    improve      = incumbent - mu[mask]
    Z            = improve / sig[mask]
    exploit      = improve * norm.cdf(Z)
    explore      = sig[mask] * norm.pdf(Z)
    values[mask] = exploit + explore
    return values

# def scipy_optimize(fn, x_s):
#     def _target(x):
#         return float(fn(x.reshape(1, -1)))
        
#     res = minimize(_target, x_s[0], bounds=[(0, None)] * x_s.shape[1])
#     return _target(res.x)

# --
# Run BO experiment

umodel = ALPACA(x_dim=train_dataset.x_dim, y_dim=1, sig_eps=0.01, hidden_dim=128, final_dim=128, activation='tanh')
umodel = umodel.cuda()

res = []
for _ in range(100):
    burnin_size         = train_kwargs['support_size']
    num_rounds          = 10
    num_bo_candidates   = 10000
    num_rand_candidates = 500
    
    x_s, y_s, _, _, fn = valid_dataset.sample_one(support_size=burnin_size, query_size=0)
    x_s_orig, y_s_orig = x_s.copy(), y_s.copy()
    
    task_x = valid_dataset.data_dict[fn['task_id']]['x']
    task_y = valid_dataset.data_dict[fn['task_id']]['y']
    y_opt  = task_y.max()
    
    # --
    # BO w/ trained model
    
    x_s, y_s  = x_s_orig.copy(), y_s_orig.copy()
    incumbent = y_s.max()
    
    model_traj = y_s.squeeze()
    for _ in range(num_rounds):
        bo_sel = np.random.choice(task_x.shape[0], num_bo_candidates)
        x_cand = task_x[bo_sel]
        
        inp = list2tensors((x_s, y_s, x_cand), cuda=model.is_cuda)
        mu, sig, _ = model(*inp)
        mu, sig = tensors2list((mu, sig), squeeze=True)
        
        mu = -mu
        
        ei = gaussian_ei(mu, sig, incumbent=incumbent)
        
        next_x = x_cand[ei.argmax()]
        next_y = task_y[bo_sel][ei.argmax()]
        
        x_s = np.vstack([x_s, next_x])
        y_s = np.vstack([y_s, next_y])
        incumbent = y_s.max()
        
        model_traj = np.hstack([model_traj, [incumbent]])
    
    # >>
    # --
    # BO w/ trained model
    
    x_s, y_s  = x_s_orig.copy(), y_s_orig.copy()
    incumbent = y_s.max()
    
    umodel_traj = y_s.squeeze()
    for _ in range(num_rounds):
        bo_sel = np.random.choice(task_x.shape[0], num_bo_candidates)
        x_cand = task_x[bo_sel]
        
        inp = list2tensors((x_s, y_s, x_cand), cuda=model.is_cuda)
        mu, sig, _ = umodel(*inp)
        mu, sig = tensors2list((mu, sig), squeeze=True)
        
        mu = -mu
        
        ei = gaussian_ei(mu, sig, incumbent=incumbent)
        
        next_x = x_cand[ei.argmax()]
        next_y = task_y[bo_sel][ei.argmax()]
        
        x_s = np.vstack([x_s, next_x])
        y_s = np.vstack([y_s, next_y])
        incumbent = y_s.max()
        
        umodel_traj = np.hstack([umodel_traj, [incumbent]])
        
    # <<
    # --
    # Random
    
    rand_sel  = np.random.choice(task_x.shape[0], num_rand_candidates, replace=False)
    rand_cand = task_x[rand_sel]
    rand_y    = task_y[rand_sel]
    rand_traj = pd.Series(rand_y.squeeze()).cummin().values
    
    res.append({
        "opt"          : y_opt,
        "rand"         : rand_traj[-1],
        "umodel_final" : umodel_traj[-1],
        "umodel_first" : umodel_traj[burnin_size],
        "model_final"  : model_traj[-1],
        "model_first"  : model_traj[burnin_size],
    })
    print(res[-1])


res = pd.DataFrame(res)
(res.model_first >= res.umodel_first).mean()
(res.model_final >= res.umodel_final).mean()

res.mean()

# I suspect all of the samples are _super_ close together






