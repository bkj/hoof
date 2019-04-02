#!/usr/bin/env python

"""
    svc_bo_clean.py
"""

from rsub import *
from matplotlib import pyplot as plt

import sys
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm, trange
from scipy.spatial.distance import cdist

import torch
from torch import nn

from hoof.dataset import SVCFileDataset
from hoof.models import ALPACA
from hoof.helpers import set_seeds, to_numpy, list2tensors, tensors2list, set_lr
from hoof.bayesopt import gaussian_ei

torch.set_num_threads(2)
set_seeds(555)

# --
# Dataset

path = 'data/topk.jl'
train_dataset = SVCFileDataset(path=path)
valid_dataset = SVCFileDataset(path=path)

# Non-overlapping tasks
num_tasks = len(train_dataset.task_ids)
num_train_tasks = 25
task_ids  = np.random.permutation(train_dataset.task_ids)
train_dataset.task_ids, valid_dataset.task_ids = task_ids[:num_train_tasks], task_ids[num_train_tasks:]

assert (train_dataset.data.param_rbf_kernel).all()

# --
# Train

print('x_dim=%d' % train_dataset.x_dim, file=sys.stderr)

model = ALPACA(input_dim=train_dataset.x_dim, output_dim=1, 
    sig_eps=0.01, hidden_dim=64, activation='relu').cuda()

opt = torch.optim.Adam(model.parameters(), lr=1e-3)

train_kwargs = {"batch_size" : 256, "query_size" : 100, "num_samples" : 30000, "mixup" : False}

train_history = []
for support_size in [10, 8, 6, 4, 10]:
    train_history += model.do_train(dataset=train_dataset, opt=opt, support_size=support_size, **train_kwargs)
    
    _ = plt.plot(train_history, c='red', label='train')
    _ = plt.yscale('log')
    _ = plt.grid()
    _ = plt.legend()
    show_plot()


valid_history = model.do_valid(dataset=valid_dataset, support_size=10, **train_kwargs)

print('final_train_loss=%f' % np.mean(train_history[-100:]), file=sys.stderr)
print('final_valid_loss=%f' % np.mean(valid_history[-100:]), file=sys.stderr)

# --
# Run BO experiment

_ = model.eval()

def random_search(x_all, y_all, num_candidates=500):
    rand_sel  = np.random.choice(x_all.shape[0], num_candidates, replace=True)
    rand_y    = y_all[rand_sel].squeeze()
    return pd.Series(rand_y).cummin().values

def alpaca_bo(model, x_all, y_all, num_rounds=20, burnin_size=2):
    burnin_sel = np.random.choice(x_all.shape[0], burnin_size, replace=False)
    x_visited, y_visited = x_all[burnin_sel], y_all[burnin_sel]
    
    traj = np.sort(y_visited.squeeze())[::-1]
    
    for _ in range(num_rounds):
        # !! Simple way to force exploration
        explore = cdist(x_all, x_visited).min(axis=-1) > 0.05
        x_cand, y_cand = x_all[explore], y_all[explore]
        
        inp = list2tensors((x_visited, y_visited, x_cand), cuda=model.is_cuda)
        mu, sig, _ = model(*inp)
        mu, sig = tensors2list((mu, sig), squeeze=True)
        
        ei = gaussian_ei(mu, sig, incumbent=y_visited.min())
        
        best_idx = ei.argmax()
        next_x, next_y = x_cand[best_idx], y_cand[best_idx]
        
        x_visited = np.vstack([x_visited, next_x])
        y_visited = np.vstack([y_visited, next_y])
        
        traj = np.hstack([traj, [y_visited.min()]])
    
    return traj


dataset = valid_dataset

res = []
for _ in trange(100):
    
    task_id      = np.random.choice(dataset.task_ids)
    x_all, y_all = dataset.data_dict[task_id]
    y_opt        = y_all.min()
    
    model_traj = alpaca_bo(model, x_all, y_all)
    rand_traj  = random_search(x_all, y_all)
    
    res.append({
        "task_id" : task_id,
        "opt"     : y_opt,
        "model"   : np.array(model_traj),
        "rand"    : np.array(rand_traj),
    })


model_adj  = [(xx['opt'] - xx['model']) / xx['opt'] for xx in res]
rand_adj   = [(xx['opt'] - xx['rand']) / xx['opt'] for xx in res]

_ = plt.plot(np.stack(model_adj).mean(axis=0), c='red')
_ = [plt.plot(xx, alpha=0.01, c='red') for xx in model_adj]

_ = plt.plot(np.stack(rand_adj).mean(axis=0), c='black')
_ = [plt.plot(xx, alpha=0.01, c='black') for xx in rand_adj]

_ = plt.xlim(0, 35)
_ = plt.ylim(0, 0.1)
show_plot()


