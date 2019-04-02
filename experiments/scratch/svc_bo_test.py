#!/usr/bin/env python

"""
    simple_example.py
"""

import sys
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm, trange

import torch
from torch import nn

from rsub import *
from matplotlib import pyplot as plt

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

# --
# Plot some samples

# >>
# --
# OpenML metadata

# import openml
# tmp = train_dataset.data[['task_id', 'mean_score', 'param_cost']]
# tmp = tmp.sort_values('param_cost').reset_index()

# tasks = []
# for task_id in tqdm(tmp.task_id.unique()):
#     try:
#         tasks.append(openml.tasks.get_task(task_id=task_id))
#     except:
#         print('error at %d' % task_id, file=sys.stderr)

# task_ids = sorted([t.task_id for t in tasks])

# for task_id in task_ids:
#     sub = tmp[tmp.task_id == task_id]
#     sub = sub.sort_values('param_cost').reset_index(drop=True)
    
#     x = np.log10(sub.param_cost)
#     y = (sub.mean_score - sub.mean_score[0]).rolling(window=50).mean()
    
#     if y.max() < 0.3:
#         _ = plt.plot(x, y, alpha=0.75)

# show_plot()
# <<

# train_dataset.task_ids = task_ids[:num_train_tasks]
# for task_id in train_dataset.task_ids:
#     X, y = train_dataset.data_dict[task_id]
#     X = X.squeeze()
#     y = y.squeeze()
#     o = np.argsort(X)
#     _ = plt.plot(X[o], y[o])

# show_plot()

# --
# Train

print('x_dim=%d' % train_dataset.x_dim, file=sys.stderr)

model = ALPACA(input_dim=train_dataset.x_dim, output_dim=1, sig_eps=0.01, hidden_dim=64, activation='relu').cuda()

train_history = []
lrs = [1e-3, 1e-4]
opt = torch.optim.Adam(model.parameters(), lr=lrs[0])

train_kwargs = {"batch_size" : 256, "support_size" : 10, "query_size" : 100, "num_samples" : 30000, "mixup" : False}

for support_size in [10, 8, 6, 4, 10]:
    train_kwargs['support_size'] = support_size
    # for lr in lrs:
        # set_lr(opt, lr)
        
    train_history  += model.do_train(dataset=train_dataset, opt=opt, **train_kwargs)
    
    _ = plt.plot(train_history, c='red', label='train')
    _ = plt.yscale('log')
    _ = plt.grid()
    _ = plt.legend()
    show_plot()


train_kwargs['support_size'] = 10
train_kwargs['mixup'] = False
valid_history = model.do_valid(dataset=valid_dataset, **train_kwargs)

print('final_train_loss=%f' % np.mean(train_history[-100:]), file=sys.stderr)
print('final_valid_loss=%f' % np.mean(valid_history[-100:]), file=sys.stderr)

# no mixup:
# final_train_loss=0.013368
# final_valid_loss=0.024562

# # --
# # Plot

# data_dict = valid_dataset.data_dict

# task_id = np.random.choice(list(data_dict.keys()))
# support_size = data_dict[task_id][0].shape[0]

# # Get all data points
# x_all, y_all, _, _, fn = valid_dataset.sample_one(support_size=support_size, query_size=0, task_id=task_id)
# x_all, y_all = x_all[np.argsort(x_all.squeeze())], y_all[np.argsort(x_all.squeeze())]

# # Get small number of points
# x_s, y_s, _, _, fn = train_dataset.sample_one(support_size=10, query_size=0, task_id=task_id)

# inp = list2tensors((x_s, y_s, x_all), cuda=model.is_cuda)
# mu, sig, _ = model(*inp)
# mu, sig = tensors2list((mu, sig), squeeze=True)

# _ = plt.plot(x_all, y_all, c='black', alpha=0.25)
# _ = plt.plot(x_all, mu)
# _ = plt.fill_between(x_all.squeeze(), mu - 1.96 * np.sqrt(sig), mu + 1.96 * np.sqrt(sig), alpha=0.2)
# _ = plt.scatter(x_s, y_s, c='red')
# show_plot()


# --
# Run BO experiment

_ = model.eval()

from scipy.spatial.distance import cdist

# umodel = ALPACA(input_dim=train_dataset.x_dim, output_dim=1, sig_eps=0.03, hidden_dim=64, activation='relu').cuda()

dataset = valid_dataset

res, trajs = [], []
for _ in trange(100):
    burnin_size         = 2 # train_kwargs['support_size'] // 2
    num_rounds          = 20
    num_bo_candidates   = 10000
    num_rand_candidates = 500
    
    x_s, y_s, _, _, fn = dataset.sample_one(support_size=burnin_size, query_size=0)
    o = y_s.squeeze().argsort()[::-1]
    x_s, y_s = x_s[o], y_s[o]
    
    x_s_orig, y_s_orig = x_s.copy(), y_s.copy()
    
    task_x, task_y = dataset.data_dict[fn['task_id']]
    # o = np.argsort(task_x, axis=0)
    # task_x, task_y = task_x[o], task_y[o]
    y_opt = task_y.min()
    
    # --
    # BO w/ trained model
    
    x_s, y_s  = x_s_orig.copy(), y_s_orig.copy()
    incumbent = y_s.min()
    
    model_traj = y_s.squeeze()
    for _ in range(num_rounds):
        bo_sel = np.random.choice(task_x.shape[0], num_bo_candidates)
        x_cand = task_x[bo_sel]
        
        # >>
        # Force exploration -- don't sample points that are too close together
        keep   = cdist(x_cand, x_s).min(axis=-1) > 0.05
        bo_sel, x_cand = bo_sel[keep], x_cand[keep]
        # <<
        
        inp = list2tensors((x_s, y_s, x_cand), cuda=model.is_cuda)
        mu, sig, _ = model(*inp)
        mu, sig = tensors2list((mu, sig), squeeze=True)
        
        ei = gaussian_ei(mu, sig, incumbent=incumbent)
        
        next_x = x_cand[ei.argmax()]
        next_y = task_y[bo_sel][ei.argmax()]
        
        x_s = np.vstack([x_s, next_x])
        y_s = np.vstack([y_s, next_y])
        incumbent = y_s.min()
        
        model_traj = np.hstack([model_traj, [incumbent]])
    
    # # --
    # # BO w/ untrained model
    
    # x_s, y_s  = x_s_orig.copy(), y_s_orig.copy()
    # incumbent = y_s.min()
    
    # umodel_traj = y_s.squeeze()
    # for _ in range(num_rounds):
    #     bo_sel = np.random.choice(task_x.shape[0], num_bo_candidates)
    #     x_cand = task_x[bo_sel]
        
    #     inp = list2tensors((x_s, y_s, x_cand), cuda=model.is_cuda)
    #     mu, sig, _ = umodel(*inp)
    #     mu, sig = tensors2list((mu, sig), squeeze=True)
        
    #     ei = gaussian_ei(mu, sig, incumbent=incumbent)
        
    #     next_x = x_cand[ei.argmax()]
    #     next_y = task_y[bo_sel][ei.argmax()]
        
    #     x_s = np.vstack([x_s, next_x])
    #     y_s = np.vstack([y_s, next_y])
    #     incumbent = y_s.min()
        
    #     umodel_traj = np.hstack([umodel_traj, [incumbent]])
    
    # --
    # Random
    
    rand_sel  = np.random.choice(task_x.shape[0], num_rand_candidates, replace=True)
    rand_cand = task_x[rand_sel]
    rand_y    = task_y[rand_sel]
    rand_traj = pd.Series(rand_y.squeeze()).cummin().values
    
    res.append({
        "task_id"      : fn['task_id'],
        "opt"          : y_opt,
        "rand"         : rand_traj[-1],
        "rand_eq"      : rand_traj[len(model_traj)],
        # "umodel_final" : umodel_traj[-1],
        # "umodel_first" : umodel_traj[burnin_size],
        "model_final"  : model_traj[-1],
        "model_first"  : model_traj[burnin_size],
    })
    trajs.append({
        "model_traj"  : (y_opt - np.array(model_traj)) / y_opt,
        # "umodel_traj" : umodel_traj,
        "rand_traj"   : (y_opt - np.array(rand_traj)) / y_opt,
    })
    # print(res[-1])

# Inspect results
res = pd.DataFrame(res)
# (res.model_first <= res.umodel_first).mean()
# (res.model_final <= res.umodel_final).mean()
(res.model_final <= res.rand_eq).mean()

(res.model_final == -1).mean()
# (res.umodel_final == -1).mean()
(res.rand_eq == -1).mean()


for t in trajs:
    _ = plt.plot(t['model_traj'], alpha=0.01, c='red')
    # _ = plt.plot(t['umodel_traj'], alpha=0.01, c='blue')
    _ = plt.plot(t['rand_traj'], alpha=0.01, c='black')


_ = plt.plot(np.stack([t['model_traj'] for t in trajs]).mean(axis=0), c='red')
# _ = plt.plot(np.stack([t['umodel_traj'] for t in trajs]).mean(axis=0), c='blue')
_ = plt.plot(np.stack([t['rand_traj'] for t in trajs]).mean(axis=0), c='black')

_ = plt.xlim(0, 35)
# _ = plt.ylim(-1.01, -0.90)
_ = plt.ylim(0, 0.1)
show_plot()




