#!/usr/bin/env python

"""
    helpers.py
"""


import torch
import numpy as np

def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    else:
        return x.cpu().detach().numpy()

def set_seeds(seed):
    _ = np.random.seed(123)
    _ = torch.manual_seed(123 + 111)
    _ = torch.cuda.manual_seed(123 + 222)


def list2tensors(xs, cuda=False):
    if not cuda:
        return list(map(torch.Tensor, xs))
    else:
        return list(map(lambda x: torch.Tensor(x).cuda(), xs))


def tensors2list(xs, squeeze=False):
    if not squeeze:
        return list(map(to_numpy, xs))
    else:
        return list(map(lambda x: to_numpy(x).squeeze(), xs))

def set_lr(opt, lr):
    for p in opt.param_groups:
        p['lr'] = lr

# --
# Metrics

class HoofMetrics:
    @staticmethod
    def mean_squared_error(y_act, y_pred):
        return float(((y_act - y_pred) ** 2).mean())