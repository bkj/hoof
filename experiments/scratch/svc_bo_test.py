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

torch.set_num_threads(1)
set_seeds(345)

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

# Plot some samples

train_dataset.data[['task_id', 'mean_score', '']]

for _ in range(10):
    x_s, y_s, _, _, _ = train_dataset.sample_one(support_size=100, query_size=0)
    _ = plt.scatter(x_s.squeeze(), y_s.squeeze())

show_plot()



# --
# Train

model = ALPACA(input_dim=train_dataset.x_dim, output_dim=1, sig_eps=0.2, hidden_dim=128, activation='relu').cuda()

train_history = []
lrs = [1e-4, 1e-5]
opt = torch.optim.Adam(model.parameters(), lr=lrs[0])

train_kwargs = {"batch_size" : 64, "support_size" : 10, "query_size" : 100, "num_samples" : 30000}

for lr in lrs:
    set_lr(opt, lr)
    
    train_history  += model.train(dataset=train_dataset, opt=opt, **train_kwargs)
    
    _ = plt.plot(train_history, c='red', label='train')
    _ = plt.yscale('log')
    _ = plt.grid()
    _ = plt.legend()
    show_plot()


valid_history = model.valid(dataset=valid_dataset, **train_kwargs)

print('final_train_loss=%f' % np.mean(train_history[-100:]), file=sys.stderr)
print('final_valid_loss=%f' % np.mean(valid_history[-100:]), file=sys.stderr)

# --
# Plot

data_dict = valid_dataset.data_dict

list(data_dict.keys())

task_id = 3492
support_size = data_dict[task_id]['x'].shape[0]
# orig_task_ids = valid_dataset.task_ids
valid_dataset.task_ids = [task_id]

x_all, y_all, _, _, fn = valid_dataset.sample_one(support_size=support_size, query_size=0)

x_all, y_all = x_all[np.argsort(x_all.squeeze())], y_all[np.argsort(x_all.squeeze())]

x_s, y_s, _, _, fn = valid_dataset.sample_one(support_size=2, query_size=0)

inp = list2tensors((x_s, y_s, x_all), cuda=model.is_cuda)
mu, sig, _ = model(*inp)
mu, sig = tensors2list((mu, sig), squeeze=True)

_ = plt.plot(x_all, y_all, c='black', alpha=0.25)
_ = plt.plot(x_all, mu)
_ = plt.fill_between(x_all.squeeze(), mu - 1.96 * np.sqrt(sig), mu + 1.96 * np.sqrt(sig), alpha=0.2)
_ = plt.scatter(x_s, y_s, c='red')
# _ = plt.xlim(*valid_dataset.x_range)
show_plot()


# --
# Run BO experiment

umodel = ALPACA(input_dim=train_dataset.x_dim, output_dim=1, sig_eps=0.01, hidden_dim=128, activation='tanh')
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
    
    rand_sel  = np.random.choice(task_x.shape[0], num_rand_candidates, replace=True)
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






