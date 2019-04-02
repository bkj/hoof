#!/usr/bin/env python

"""
    dataset.py
"""

import sys
import json
import numpy as np
import pandas as pd
from copy import copy

import sklearn
import sklearn.compose
import sklearn.impute
import sklearn.feature_selection

# --
# Base Dataset

class _BaseDataset:
    def __init__(self, popsize=None, seed=None):
        
        if not seed:
            seed = np.random.choice(2 ** 15)
            
        self.seed = seed
        self.rng  = np.random.RandomState(seed)
        
        self.popsize = popsize
        if self.popsize is not None:
            print('setting population to %d functions' % self.popsize, file=sys.stderr)
            self.fn_pop = [self.sample_fn() for _ in range(self.popsize)]
            
    def set_seed(self, seed):
        self.seed = seed
        self.rng  = np.random.RandomState(seed)
    
    def _sample_fn_wrapper(self):
        if self.popsize is None:
            return self.sample_fn()
        else:
            return self.fn_pop[self.rng.choice(self.popsize)]
    
    def sample_x(self, n):
        raise NotImplemented
    
    def sample_fn(self, *args, **kwargs):
        raise NotImplemented
    
    def sample_one(self, support_size, query_size):
        x_support = self.sample_x(n=support_size)
        x_query   = self.sample_x(n=query_size)
        
        fn = self._sample_fn_wrapper()
        
        y_support = fn(x_support)
        y_query   = fn(x_query)
        
        return x_support, y_support, x_query, y_query, fn
    
    def sample_batch(self, support_size, query_size, batch_size):
        samples = [self.sample_one(support_size=support_size, query_size=query_size) for _ in range(batch_size)]
        x_support, y_support, x_query, y_query, fn = list(zip(*samples))
        
        return (
            np.stack(x_support),
            np.stack(y_support),
            np.stack(x_query),
            np.stack(y_query),
            fn,
        )



class SinusoidDataset(_BaseDataset):
    def __init__(self, 
        amp_range=[0.1, 5.0],
        phase_range=[0, 3.14], 
        freq_range=[0.999, 1.0],
        x_range=[-5, 5],
        noise_std=0.0,
        **kwargs
    ):
        
        self.amp_range   = amp_range
        self.phase_range = phase_range
        self.freq_range  = freq_range
        self.x_range     = x_range
        self.noise_std   = noise_std
        
        super().__init__(**kwargs)
    
    def sample_x(self, n):
        return self.rng.uniform(*self.x_range, (n, 1))
    
    def sample_fn(self):
        amp   = self.rng.uniform(*self.amp_range)
        phase = self.rng.uniform(*self.phase_range)
        freq  = self.rng.uniform(*self.freq_range)
        
        def _fn(x):
            tmp = amp * np.sin(freq * x + phase)
            if self.noise_std > 0:
                tmp += np.random.normal(0, self.noise_std, tmp.shape)
            
            return tmp
        
        return _fn


class NoisySinusoidDataset(_BaseDataset):
    def __init__(self, 
        amp_range=[0.1, 5.0],
        phase_range=[0, 3.14], 
        freq_range=[0.999, 1.0],
        noise_range=[0.0, 0.5],
        x_range=[-5, 5],
        
        **kwargs
    ):
        
        self.amp_range   = amp_range
        self.phase_range = phase_range
        self.freq_range  = freq_range
        self.x_range     = x_range
        self.noise_range = noise_range
        
        super().__init__(**kwargs)
    
    def sample_x(self, n):
        return self.rng.uniform(*self.x_range, (n, 1))
    
    def sample_fn(self):
        amp   = self.rng.uniform(*self.amp_range)
        phase = self.rng.uniform(*self.phase_range)
        freq  = self.rng.uniform(*self.freq_range)
        noise = self.rng.uniform(*self.noise_range)
        
        def _fn(x):
            tmp = amp * np.sin(freq * x + phase)
            tmp += np.random.normal(0, noise, tmp.shape)
            return tmp
        
        return _fn


class SmileFrownDataset(_BaseDataset):
    def __init__(self, x_range=[-3, 3], **kwargs):
        self.x_range = x_range
        
        super().__init__(**kwargs)
    
    def sample_x(self, n):
        return self.rng.uniform(*self.x_range, (n, 1))
    
    def sample_fn(self):
        c = np.random.choice((-1, 1))
        def _fn(x):
            return c * (x ** 2) - c
        
        return _fn


class LineDataset(_BaseDataset):
    def __init__(self, x_range=[0, 1], **kwargs):
        self.x_range = x_range
        
        super().__init__(**kwargs)
    
    def sample_x(self, n):
        return self.rng.uniform(*self.x_range, (n, 1))
    
    def sample_fn(self):
        def _fn(x):
            return x
        
        return _fn


class PowerDataset(_BaseDataset):
    def __init__(self, x_range=[0, 1], **kwargs):
        self.x_range = x_range
        
        super().__init__(**kwargs)
    
    def sample_x(self, n):
        return self.rng.uniform(*self.x_range, (n, 1))
    
    def sample_fn(self):
        # >>
        # Harder
        
        # c = self.rng.uniform(1, 5)
        # p = self.rng.choice([2, 3, 100])
        
        # def _fn(x):
        #     return c * x ** p
        
        # --
        # Simpler
        
        p = self.rng.uniform(1, 5)
        def _fn(x):
            return x ** p
        # <<
        
        return _fn


class QuadraticDataset(_BaseDataset):
    def __init__(self, x_range=[-2, 2], x_dim=1, **kwargs):
        self.x_range = x_range
        self.x_dim   = x_dim
        
        super().__init__(**kwargs)
    
    def sample_x(self, n):
        return self.rng.uniform(*self.x_range, (n, self.x_dim))
    
    def sample_fn(self):
        alpha = self.rng.uniform(0.1, 10, 3)
        def _fn(x):
            assert len(x.shape) > 1
            return (
                0.5 * alpha[0] * (x ** 2).mean(axis=-1, keepdims=True) +
                alpha[1] * x.mean(axis=-1, keepdims=True) +
                alpha[2]
            )
        
        return _fn


class SVCFileDataset(_BaseDataset):
    def __init__(self, path, **kwargs):
        
        self.data     = self._load_data(path)
        self.task_ids = list(set(self.data.task_id))
        
        self.data_dict = {}
        for task_id in self.task_ids:
            sub = self.data[self.data.task_id == task_id]
            
            X = np.vstack(sub.Xf.values)
            y = sub.mean_score.values
            
            # Scale to [0, 1]
            # y = (y - y.min()) / (y.max() - y.min())
            
            y = y.reshape(-1, 1)
            
            # Minimize (by convention)
            y = -1 * y
            
            self.data_dict[task_id] = (X, y)
        
        self.x_cols = ['param_cost', 'param_degree', 'param_gamma', 'param_kernel']
        
        super().__init__(**kwargs)
    
    def _load_data(self, path):
        data = [json.loads(xx) for xx in open(path).read().splitlines()]
        data = [{
            "task_id"             : xx['task_id'],
            "mean_score"          : np.mean(xx['scores']),
            "all_scores"          : xx['scores'],
            
            "param_rbf_kernel"    : int(xx['params'].get('kernel', None) is None),
            "param_linear_kernel" : int(xx['params'].get('kernel', None) == 'linear'),
            "param_poly_kernel"   : int(xx['params'].get('kernel', None) == 'polynomial'),
            "param_cost"          : xx['params'].get('cost', None),
            "param_gamma"         : xx['params'].get('gamma', None),
            "param_degree"        : xx['params'].get('degree', None),
        } for xx in data]
        
        data = pd.DataFrame(data, columns=list(data[0].keys()))
        data = data.sort_values('param_cost').reset_index(drop=True)
        
        for c in ['param_cost', 'param_gamma', 'param_degree']:
            data[c] = data[c].astype(np.float64)
        
        data.param_degree = data.param_degree.fillna(1)
        data.param_cost   = np.log10(data.param_cost)
        data.param_gamma  = np.log10(data.param_gamma)
        
        param_degree_cols = []
        for v in data.param_degree.unique():
            data['param_degree_%d' % v] = data.param_degree == v
            param_degree_cols.append('param_degree_%d' % v)
        
        # >>
        print('FileDataset: rbf kernel only', file=sys.stderr)
        data = data[data.param_rbf_kernel.astype(bool)]
        Xf   = data[['param_cost', 'param_gamma']].values
        # <<
        
        self.x_dim = Xf.shape[1]
        data['Xf'] = [tuple(xx) for xx in Xf]
        
        return data
    
    def sample_one(self, support_size, query_size, task_id=None):
        if task_id is None:
            task_id = np.random.choice(self.task_ids)
        
        X, y = self.data_dict[task_id]
        
        sel  = np.random.choice(X.shape[0], support_size + query_size, replace=False)
        X, y = X[sel], y[sel]
        
        X_support, X_query = X[:support_size], X[support_size:]
        y_support, y_query = y[:support_size], y[support_size:]
        
        return X_support, y_support, X_query, y_query, {"task_id" : task_id}




