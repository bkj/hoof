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
