#!/usr/bin/env python

"""
    hoof/bayesopt.py
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

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


def scipy_optimize(fn, p):
    def _target(x):
        return float(fn(x.reshape(1, -1)))
        
    res = minimize(_target, p, bounds=[(0, None)] * p.shape[1])
    return res.x, res.fun


