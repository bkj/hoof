#!/usr/bin/env python

"""
    dataset.py
"""

import numpy as np
from copy import copy

def _make_sin_func(amp, phase, freq, noise_std, rng):
    def _f(x, noise=True):
        y = amp * np.sin(freq * x + phase)
        if noise:
            y += np.random.normal(0, noise_std, y.shape)
        
        return y
    
    return _f

class SinusoidDataset:
    def __init__(self, amp_range=[0.1, 5.0], phase_range=[0, 3.14], 
        freq_range=[0.999, 1.0], x_range=[-5, 5], sig_eps=0.02):
        
        self.amp_range   = amp_range
        self.phase_range = phase_range
        self.freq_range  = freq_range
        self.x_range     = x_range
        self.noise_std   = np.sqrt(sig_eps)
        
    def sample(self, n_funcs, train_samples, test_samples, rng=None):
        if rng is None:
            rng = np.random
        
        x_c = rng.uniform(*self.x_range, (n_funcs, train_samples, 1))
        x   = rng.uniform(*self.x_range, (n_funcs, test_samples, 1))
        
        y_c = np.zeros((n_funcs, train_samples, 1))
        y   = np.zeros((n_funcs, test_samples, 1))
        
        funcs = []
        for i in range(n_funcs):
            
            amp   = rng.uniform(*self.amp_range)
            phase = rng.uniform(*self.phase_range)
            freq  = rng.uniform(*self.freq_range)
            
            _f     = _make_sin_func(amp, phase, freq, self.noise_std, rng)
            y_c[i] = _f(x_c[i])
            y[i]   = _f(x[i])
            
            funcs.append(copy(_f))
        
        return x_c, y_c, x, y, funcs

# --
# Make pow func

def _make_pow_func(p, rng):
    def _f(x):
        y = x ** p
        return y
    
    return _f

class PowerDataset:
    def __init__(self, x_range=[0, 1]):
        self.x_range = x_range
    
    def sample(self, n_funcs, train_samples, test_samples, rng=None):
        if rng is None:
            rng = np.random
        
        x_c = rng.uniform(*self.x_range, (n_funcs, train_samples, 1))
        x   = rng.uniform(*self.x_range, (n_funcs, test_samples, 1))
        
        y_c = np.zeros((n_funcs, train_samples, 1))
        y   = np.zeros((n_funcs, test_samples, 1))
        
        funcs = []
        for i in range(n_funcs):
            
            p = rng.uniform(1, 5)
            
            _f     = _make_pow_func(p, rng)
            y_c[i] = _f(x_c[i])
            y[i]   = _f(x[i])
            
            funcs.append(copy(_f))
        
        return x_c, y_c, x, y, funcs

# --

def _make_quadratic_func(xo, yo, rng):
    def _f(x):
        y = (x + xo) ** 2 + yo
        return y
    
    return _f

class QuadraticDataset:
    def __init__(self, x_range=[-2, -2]):
        self.x_range = x_range
    
    def sample(self, n_funcs, train_samples, test_samples, rng=None):
        if rng is None:
            rng = np.random
        
        x_c = rng.uniform(*self.x_range, (n_funcs, train_samples, 1))
        x   = rng.uniform(*self.x_range, (n_funcs, test_samples, 1))
        
        y_c = np.zeros((n_funcs, train_samples, 1))
        y   = np.zeros((n_funcs, test_samples, 1))
        
        funcs = []
        for i in range(n_funcs):
            
            xo = rng.uniform(-1, 1)
            yo = rng.uniform(-1, 1)
            
            _f     = _make_quadratic_func(xo, yo, rng)
            y_c[i] = _f(x_c[i])
            y[i]   = _f(x[i])
            
            funcs.append(copy(_f))
        
        return x_c, y_c, x, y, funcs


# --

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

def _make_sgd_func(rng, dim=5, nobs=1000, pos_mean=0, pos_std=1, neg_mean=1, neg_std=1):
    warnings.filterwarnings("ignore", category=FutureWarning)
    pos = rng.normal(pos_mean, pos_std, (nobs, dim))
    neg = rng.normal(neg_mean, neg_std, (nobs, dim))
    X   = np.vstack([pos, neg]).astype(float)
    X /= np.sqrt((X ** 2).sum(axis=-1, keepdims=True))
    y   = np.hstack([
        np.ones(pos.shape[0]),
        np.zeros(neg.shape[0])
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5)
    
    def _f(alphas):
        accs = []
        for alpha in alphas:
            sgd = SGDClassifier(learning_rate='constant', eta0=10 ** float(alpha), random_state=111)
            sgd = sgd.fit(X_train, y_train)
            preds = sgd.predict(X_test)
            acc = (y_test == preds).mean()
            accs.append(acc)
        
        return np.array(accs).reshape(alphas.shape)
    
    return _f

class SGDDataset:
    def __init__(self, x_range=[-5, 0]):
        self.x_range = x_range
    
    def sample(self, n_funcs, train_samples, test_samples, rng=None):
        if rng is None:
            rng = np.random
        
        x_c = rng.uniform(*self.x_range, (n_funcs, train_samples, 1))
        x   = rng.uniform(*self.x_range, (n_funcs, test_samples, 1))
        
        y_c = np.zeros((n_funcs, train_samples, 1))
        y   = np.zeros((n_funcs, test_samples, 1))
        
        funcs = []
        for i in range(n_funcs):
            _f     = _make_sgd_func(rng)
            y_c[i] = _f(x_c[i])
            y[i]   = _f(x[i])
            
            funcs.append(copy(_f))
        
        return x_c, y_c, x, y, funcs

# --

from time import time
from joblib import Parallel, delayed

class CacheDataset:
    def __init__(self, dataset, n_batches, n_jobs=32, **kwargs):
        self.n_batches = n_batches
        self.offset    = 0
        
        # self._cache = [None] * n_batches
        # for idx in range(n_batches):
        #     self._cache[idx] = dataset.sample(**kwargs)
        #     # Make this parallel
        
        jobs = [delayed(dataset.sample)(rng=np.random.RandomState(i), **kwargs) for i in range(n_batches)]
        self._cache = Parallel(n_jobs=n_jobs)(jobs)
        
    def sample(self, *args, **kwargs):
        out = self._cache[self.offset]
        self.offset = (self.offset + 1) % self.n_batches
        return out
