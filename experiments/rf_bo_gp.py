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
from joblib import Parallel, delayed

import torch
from torch import nn

from hoof.dataset import RFFileDataset
from hoof.models import ALPACA
from hoof.helpers import set_seeds, to_numpy, list2tensors, tensors2list, set_lr
from hoof.bayesopt import gaussian_ei

torch.set_num_threads(2)
set_seeds(222)

# --
# Dataset

path = 'data/openml_lt10k.jl'
# path = '/home/bjohnson/projects/exline/results/rf_bulk/exline1/2048/openml_lt10k.jl'
train_dataset = RFFileDataset(path=path)
valid_dataset = RFFileDataset(data=train_dataset.data)

# Non-overlapping tasks
num_tasks       = len(train_dataset.task_ids)
num_train_tasks = min(50, int(num_tasks * 0.8))
task_ids = np.random.permutation(train_dataset.task_ids)
train_dataset.task_ids, valid_dataset.task_ids = task_ids[:num_train_tasks], task_ids[num_train_tasks:]

# # >>

# from skopt import Optimizer

# df = valid_dataset.data

# gp_param_cols = [
#     "param_class_weight",
#     "param_estimator",
#     "param_max_features",
#     "param_min_samples_leaf",
#     "param_min_samples_split",
# ]
# X_gp = df[['task_id', 'valid_score'] + gp_param_cols]
# X_gp.valid_score = - X_gp.valid_score

# X_gp.param_class_weight = X_gp.param_class_weight.fillna('none')

# dimensions = []
# for c in gp_param_cols:
#     if X_gp[c].dtype == np.object_:
#         uvals = list(np.unique(X_gp[c]))
#         dimensions.append(uvals)
#     else:
#         dimensions.append((
#             0.99 * X_gp[c].min(),
#             1.01 * X_gp[c].max()
#         ))


# def cummin(x):
#     return pd.Series(x).cummin().values

# def _run_one_gp(task_id, sub, max_steps=40):
#     X_cand = [list(xx) for xx in sub[gp_param_cols].values]
#     lookup = dict(zip(map(tuple, X_cand), sub.valid_score))
    
#     y_opt  = sub.valid_score.min()
    
#     opt = Optimizer(
#         dimensions=dimensions,
#         base_estimator="GP",
#         n_initial_points=3,
#         acq_func="EI",
#         acq_optimizer="sampling"
#     )
    
#     opt._X_cand = X_cand
    
#     traj = []
#     t    = time()
#     for iteration in range(max_steps):
#         if len(traj) and (np.min(traj) == y_opt):
#             # If we've already found the best, just exit early
#             next_y = np.min(traj)
#         else:
#             next_x = opt.ask()
#             opt._X_cand.remove(next_x) # !! Never revisit
#             next_y = lookup[tuple(next_x)]
#             _ = opt.tell(next_x, next_y)
        
#         traj.append(next_y)
    
#     return {
#         "task_id"    : task_id,
#         "opt"        : y_opt,
#         "gp"         : cummin(traj),
#         "gp_elapsed" : time() - t
#     }


# task_ids = valid_dataset.task_ids

# jobs = []
# for _ in range(180):
#     task_id  = np.random.choice(task_ids)
#     sub      = X_gp[X_gp.task_id == task_id]
#     jobs.append(delayed(_run_one_gp)(task_id, sub))

# gp_res = Parallel(n_jobs=60, backend='multiprocessing', verbose=10)(jobs)

# # <<

# --
# Train

print('x_dim=%d' % train_dataset.x_dim, file=sys.stderr)

model = ALPACA(
    input_dim=train_dataset.x_dim,
    output_dim=1,
    sig_eps=1e-3,
    hidden_dim=128,
    activation='ReLU'
)

opt = torch.optim.Adam(model.parameters(), lr=1e-3)

train_kwargs = {
    "batch_size"  : 256,
    "query_size"  : 120, 
    "num_samples" : 50000, 
}

model = model.cuda()
train_mse_history, train_nll_history = [], []
for lr in [1e-3, 1e-4]:
    set_lr(opt, lr)
    mse_hist, nll_hist = model.do_train(
        dataset=train_dataset,
        opt=opt, 
        support_size=[1, 7],
        **train_kwargs
    )
    
    train_mse_history += mse_hist
    train_nll_history += nll_hist

_ = plt.plot(train_nll_history, c='blue', label='train')
_ = plt.grid()
_ = plt.legend()
show_plot()

valid_mse_history, valid_nll_history = model.do_valid(
    dataset=valid_dataset, 
    support_size=10,
    batch_size=train_kwargs['batch_size'],
    query_size=train_kwargs['query_size'],
    num_samples=train_kwargs['num_samples'] // 10,
)

print('final_train_mse=%f' % np.mean(train_mse_history[-100:]), file=sys.stderr)
print('final_train_nll=%f' % np.mean(train_nll_history[-100:]), file=sys.stderr)
print('final_valid_mse=%f' % np.mean(valid_mse_history), file=sys.stderr)
print('final_valid_nll=%f' % np.mean(valid_nll_history), file=sys.stderr)

raise Exception

# --
# Run BO experiment

def random_search(x_all, y_all, num_candidates=1000):
    num_candidates = min(num_candidates, x_all.shape[0])
    rand_sel  = np.random.choice(x_all.shape[0], num_candidates, replace=False)
    rand_y    = y_all[rand_sel].squeeze()
    return pd.Series(rand_y).cummin().values


from copy import deepcopy
def alpaca_bo(model, x_all, y_all, num_rounds=20, burnin_size=2, explore_eps=0.001, acq='ei', adjust_alpha=False):
    model = deepcopy(model)
    
    burnin_sel = np.random.choice(x_all.shape[0], burnin_size, replace=False)
    x_visited, y_visited = x_all[burnin_sel], y_all[burnin_sel]
    
    traj = np.sort(y_visited.squeeze())[::-1]
    
    for iteration in range(num_rounds - burnin_size):
        
        if adjust_alpha and (iteration in [8, 16, 32, 64, 128, 256, 512]):
            model.blr.alpha /= 2
        
        # !! Simple way to force exploration
        explore = cdist(x_all, x_visited).min(axis=-1) > explore_eps
        if explore.any():
            x_cand, y_cand = x_all[explore], y_all[explore]
            
            inp = list2tensors((x_visited, y_visited, x_cand), cuda=model.is_cuda)
            mu, sig, _ = model(*inp)
            mu, sig = tensors2list((mu, sig), squeeze=True)
            
            # !! Could do "epsilon-greedy" type thing here
            
            if acq == 'ei':
                # Expected improvement
                ei       = gaussian_ei(mu, sig, incumbent=y_visited.min())
                best_idx = ei.argmax()
            elif acq == 'ucb':
                # UCB
                beta = 2
                ucb  = mu - beta * sig
                best_idx = ucb.argmin()
            
            next_x, next_y = x_cand[best_idx], y_cand[best_idx]
            
            x_visited = np.vstack([x_visited, next_x])
            y_visited = np.vstack([y_visited, next_y])
            
        traj = np.hstack([traj, [y_visited.min()]])
    
    return traj


def _run_one_bo(task_id, x_all, y_all):
    y_opt = y_all.min()
    
    model_traj = alpaca_bo(model, x_all, y_all, num_rounds=500)
    alpha_traj = alpaca_bo(model, x_all, y_all, num_rounds=500, adjust_alpha=True)
    rand_traj  = random_search(x_all, y_all)
    
    return {
        "task_id" : task_id,
        "opt"     : y_opt,
        "model"   : np.array(model_traj),
        "alpha"   : np.array(alpha_traj),
        "rand"    : np.array(rand_traj),
    }

# --
# Parallel eval

np.random.seed(123)

model = model.cpu()
model = model.eval()

dataset = valid_dataset

# !! This seems to help, especially at the end of training
# model.blr.sig_eps.data *= 10 # ?? This seems to help sometimes

jobs = []
num_valid_tasks = len(dataset.task_ids)
for _ in range(6):
    for task_id in dataset.task_ids:
        x_all, y_all = dataset.data_dict[task_id]
        job = delayed(_run_one_bo)(task_id, x_all, y_all)
        jobs.append(job)

jobs = [jobs[i] for i in np.random.permutation(len(jobs))]
print('len(jobs)=%d' % len(jobs), file=sys.stderr)

res = Parallel(n_jobs=60, verbose=10)(jobs)

# --
# Plot

df = valid_dataset.data
def agg_and_plot(res, name, c='black', mode='abs'):
    if mode == 'percentile':
        adj    = np.stack([1 - (r[name].reshape(-1, 1) <= - df[df.task_id == r['task_id']].valid_score.values).mean(axis=-1) for r in res])
    elif mode == 'abs':
        adj    = np.stack([(xx['opt'] - xx[name]) / xx['opt'] for xx in res])
    else:
        raise Exception
    
    mean   = np.mean(adj, axis=0)
    median = np.median(adj, axis=0)
    _ = plt.plot(mean, c=c, linewidth=3, label='mean(%s)' % name)
    _ = plt.plot(median, c=c, alpha=0.5, label='median(%s)' % name)
    _ = [plt.plot(xx, alpha=0.01, c=c) for xx in adj]
    
    return adj

# Plot abs
model_adj = agg_and_plot(res, name='model', c='red', mode='abs')
alpha_adj = agg_and_plot(res, name='alpha', c='purple', mode='abs')
rand_adj  = agg_and_plot(res, name='rand', c='blue', mode='abs')
gp_adj    = agg_and_plot(gp_res, name='gp', c='green', mode='abs')

_ = plt.legend()
_ = plt.ylim(1e-5, 1)
_ = plt.title('ALPACA-BO vs RANDOM')
_ = plt.xlabel('iteration')
_ = plt.ylabel('(opt_score - score) / opt_score')
_ = plt.yscale('log')
_ = plt.xscale('log')
_ = plt.grid()
show_plot()

# Plot percentile
model_adj = agg_and_plot(res, name='model', c='red', mode='percentile')
alpha_adj = agg_and_plot(res, name='alpha', c='purple', mode='percentile')
rand_adj  = agg_and_plot(res, name='rand', c='blue', mode='percentile')
gp_adj    = agg_and_plot(gp_res, name='gp', c='green', mode='percentile')

_ = plt.legend()
_ = plt.title('ALPACA-BO vs RANDOM')
_ = plt.xlabel('iteration')
_ = plt.ylabel('percentile')
_ = plt.xscale('log')
_ = plt.yscale('log')
_ = plt.grid()
show_plot()


