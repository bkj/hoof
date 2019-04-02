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

from hoof.dataset import RFFileDataset
from hoof.models import ALPACA
from hoof.helpers import set_seeds, to_numpy, list2tensors, tensors2list, set_lr
from hoof.bayesopt import gaussian_ei

torch.set_num_threads(2)
set_seeds(111)

# --
# Dataset

path = 'data/openml_l1t10k.jl'
train_dataset = RFFileDataset(path=path)
valid_dataset = RFFileDataset(data=train_dataset.data)

# Non-overlapping tasks
num_tasks = len(train_dataset.task_ids)
num_train_tasks = 50
task_ids  = np.random.permutation(train_dataset.task_ids)
train_dataset.task_ids, valid_dataset.task_ids = task_ids[:num_train_tasks], task_ids[num_train_tasks:]

# --
# Train

print('x_dim=%d' % train_dataset.x_dim, file=sys.stderr)

model = ALPACA(input_dim=train_dataset.x_dim, output_dim=1, 
    sig_eps=0.001, hidden_dim=64, activation='relu')

opt = torch.optim.Adam(model.parameters(), lr=1e-3)

train_kwargs = {
    "batch_size"  : 256,
    "query_size"  : 100, 
    "num_samples" : 30000, 
}

model = model.cuda()
train_mse_history = []
for lr in [1e-3, 1e-4]:
    set_lr(opt, lr)
    for support_size in [5, 10, 20]:
        mse_hist, _ = model.do_train(dataset=train_dataset, opt=opt, 
            support_size=support_size, **train_kwargs)
        
        train_mse_history += mse_hist

_ = plt.plot(train_mse_history, c='red', label='train')
_ = plt.yscale('log')
_ = plt.grid()
_ = plt.legend()
show_plot()

valid_mse_history, _ = model.do_valid(dataset=valid_dataset, support_size=10, **train_kwargs)

print('final_train_loss=%f' % np.mean(train_mse_history[-100:]), file=sys.stderr)
print('final_valid_loss=%f' % np.mean(valid_mse_history), file=sys.stderr)

# --
# Run BO experiment

def random_search(x_all, y_all, num_candidates=1000):
    rand_sel  = np.random.choice(x_all.shape[0], num_candidates, replace=True)
    rand_y    = y_all[rand_sel].squeeze()
    return pd.Series(rand_y).cummin().values

def alpaca_bo(model, x_all, y_all, num_rounds=20, burnin_size=2, explore_eps=0.05):
    burnin_sel = np.random.choice(x_all.shape[0], burnin_size, replace=False)
    x_visited, y_visited = x_all[burnin_sel], y_all[burnin_sel]
    
    traj = np.sort(y_visited.squeeze())[::-1]
    
    for _ in range(num_rounds - burnin_size):
        # !! Simple way to force exploration
        explore = cdist(x_all, x_visited).min(axis=-1) > explore_eps
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

# >>
# Serial eval

# _ = model.cuda()
# _ = model.eval()
# dataset = valid_dataset
# res = []
# for _ in trange(100):
    
#     task_id      = np.random.choice(dataset.task_ids)
#     x_all, y_all = dataset.data_dict[task_id]
#     y_opt        = y_all.min()
    
#     model_traj = alpaca_bo(model, x_all, y_all, num_rounds=20)
#     rand_traj  = random_search(x_all, y_all)
    
#     res.append({
#         "task_id" : task_id,
#         "opt"     : y_opt,
#         "model"   : np.array(model_traj),
#         "rand"    : np.array(rand_traj),
#     })

# --
# Parallel eval

from joblib import Parallel, delayed

def _run_one_bo(task_id, x_all, y_all):
    y_opt      = y_all.min()
    model_traj = alpaca_bo(model, x_all, y_all, num_rounds=500)
    rand_traj  = random_search(x_all, y_all)
    
    return {
        "task_id" : task_id,
        "opt"     : y_opt,
        "model"   : np.array(model_traj),
        "rand"    : np.array(rand_traj),
    }

model   = model.cpu()
model   = model.eval()
dataset = valid_dataset

jobs = []
for _ in range(120):
    task_id      = np.random.choice(dataset.task_ids)
    x_all, y_all = dataset.data_dict[task_id]
    
    job = delayed(_run_one_bo)(task_id, x_all, y_all)
    jobs.append(job)

res = Parallel(n_jobs=60, verbose=10, backend='multiprocessing')(jobs)

# <<

model_adj  = [(xx['opt'] - xx['model']) / xx['opt'] for xx in res]
rand_adj   = [(xx['opt'] - xx['rand']) / xx['opt'] for xx in res]

model_mean = np.mean(np.stack(model_adj), axis=0)
rand_mean  = np.mean(np.stack(rand_adj), axis=0)

model_median = np.median(np.stack(model_adj), axis=0)
rand_median  = np.median(np.stack(rand_adj), axis=0)

_ = plt.plot(model_mean, c='red', linewidth=3, label='mean(model)')
_ = plt.plot(model_median, c='red', alpha=0.5, label='median(model)')
_ = [plt.plot(xx, alpha=0.01, c='red') for xx in model_adj]

_ = plt.plot(rand_mean, c='blue', linewidth=3, label='mean(rand)')
_ = plt.plot(rand_median, c='blue', alpha=0.5, label='median(rand)')
_ = [plt.plot(xx, alpha=0.01, c='blue') for xx in rand_adj]

_ = plt.legend()
_ = plt.yscale('log')
show_plot()

# On some, scores very good right away
# _ = plt.plot(model_median, c='red', label='median(model)')
# _ = plt.plot(rand_median, c='blue', label='median(rand)')
# _ = plt.legend()
# _ = plt.ylim(0.0, 0.05)
# _ = plt.xlim(0, 30)
# show_plot()

# How long does it take rand to catch model?
model_mean.shape[0]
rand_wins = rand_mean < model_mean[-1]
if not rand_wins.any():
    print('rand never wins!', file=sys.stderr)
else:
    print('rand wins after %d rounds' % np.where()[0][0], file=sys.stderr)
