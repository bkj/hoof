#!/usr/bin/env python

"""
    dataset.py
"""

import sys
import numpy as np
from copy import copy

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
    def __init__(self, x_range=[-1, 1], x_dim=1, **kwargs):
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


# # --
# # Gradient descent on absolute value function

# def _make_abs_func(slope, start):
#     def _f(x):
#         return slope * np.abs(x)
        
#     def _grad(x):
#         return slope * np.sign(x)
        
#     def _do_run(alpha, num_steps=10):
#         p = start
#         for _ in range(num_steps):
#             p -= (10 ** alpha) * _grad(p)
        
#         return _f(p)
    
#     return _do_run


# class AbsValDataset:
#     def __init__(self, x_range=[-3, -0.5], slope_range=[0.5, 5], start_range=[-2, 2], popsize=None, seed=None):
        
#         self.x_range     = x_range
#         self.slope_range = slope_range
#         self.start_range = start_range
        
#         if not seed:
#             seed = np.random.choice(2 ** 15)
        
#         self.seed = seed
#         self.rng  = np.random.RandomState(seed)
        
#         self.popsize = popsize
#         if self.popsize is not None:
#             self._params = [self._sample_params() for _ in range(self.popsize)]
    
#     def set_seed(self, seed):
#         self.seed = seed
#         self.rng  = np.random.RandomState(seed)
    
#     def _sample_params(self):
#         slope = self.rng.uniform(*self.slope_range)
#         start = self.rng.uniform(*self.start_range)
#         return slope, start
    
#     def _get_params(self):
#         if self.popsize is None:
#             return self._sample_params()
#         else:
#             return self._params[self.rng.choice(self.popsize)]
    
#     def sample(self, n_funcs, train_samples, test_samples):
        
#         x_c = self.rng.uniform(*self.x_range, (n_funcs, train_samples, 1))
#         x   = self.rng.uniform(*self.x_range, (n_funcs, test_samples, 1))
        
#         y_c = np.zeros((n_funcs, train_samples, 1))
#         y   = np.zeros((n_funcs, test_samples, 1))
        
#         funcs = []
#         for i in range(n_funcs):
#             params = self._get_params()
#             _f     = _make_abs_func(*params)
#             y_c[i] = _f(x_c[i])
#             y[i]   = _f(x[i])
#             funcs.append(copy(_f))
        
#         return x_c, y_c, x, y, funcs

# # --
# # Make pow func

# def _make_pow_func(c, p):
#     def _f(x):
#         y = c * x ** p
#         return y
    
#     return _f

# class PowerDataset:
#     def __init__(self, x_range=[0, 1]):
#         self.x_range = x_range
    
#     def sample(self, n_funcs, train_samples, test_samples, rng=None):
#         if rng is None:
#             rng = np.random
        
#         x_c = rng.uniform(*self.x_range, (n_funcs, train_samples, 1))
#         x   = rng.uniform(*self.x_range, (n_funcs, test_samples, 1))
        
#         y_c = np.zeros((n_funcs, train_samples, 1))
#         y   = np.zeros((n_funcs, test_samples, 1))
        
#         funcs = []
#         for i in range(n_funcs):
            
#             c = rng.uniform(1, 5)
#             p = np.random.choice([2, 3, 100])
            
#             _f     = _make_pow_func(c, p)
#             y_c[i] = _f(x_c[i])
#             y[i]   = _f(x[i])
            
#             funcs.append(copy(_f))
        
#         return x_c, y_c, x, y, funcs

# # --
# # Parameterized Quadratic

# def _make_pquad_func(alpha):
#     def _f(x):
#         return (
#             0.5 * alpha[0] * (x ** 2) +
#             alpha[1] * x +
#             alpha[2]
#         )
    
#     return _f


# class PQuadDataset(_BaseDataset):
#     def __init__(self, x_range=[-5, 5], x_dim=1, **kwargs):
#         self.x_range = x_range
#         self.x_dim   = x_dim
        
#         super().__init__(**kwargs)
    
#     def _sample_params(self):
#         return self.rng.uniform(0.1, 10, 3)
    
#     def sample(self, n_funcs, train_samples, test_samples):
        
#         x_c = self.rng.uniform(*self.x_range, (n_funcs, train_samples, self.x_dim))
#         x   = self.rng.uniform(*self.x_range, (n_funcs, test_samples, self.x_dim))
        
#         y_c = np.zeros((n_funcs, train_samples, 1))
#         y   = np.zeros((n_funcs, test_samples, 1))
        
#         funcs = []
#         for i in range(n_funcs):
            
#             alpha  = self._get_params()
#             _f     = _make_pquad_func(alpha)
#             y_c[i] = _f(x_c[i])
#             y[i]   = _f(x[i])
            
#             funcs.append(copy(_f))
        
#         return x_c, y_c, x, y, funcs


# # # --

# # def _make_quadratic_func(xo, yo, rng):
# #     def _f(x):
# #         y = (x + xo) ** 2 + yo
# #         return y
    
# #     return _f

# # class QuadraticDataset:
# #     def __init__(self, x_range=[-2, -2]):
# #         self.x_range = x_range
    
# #     def sample(self, n_funcs, train_samples, test_samples, rng=None):
# #         if rng is None:
# #             rng = np.random
        
# #         x_c = rng.uniform(*self.x_range, (n_funcs, train_samples, 1))
# #         x   = rng.uniform(*self.x_range, (n_funcs, test_samples, 1))
        
# #         y_c = np.zeros((n_funcs, train_samples, 1))
# #         y   = np.zeros((n_funcs, test_samples, 1))
        
# #         funcs = []
# #         for i in range(n_funcs):
            
# #             xo = rng.uniform(-1, 1)
# #             yo = rng.uniform(-1, 1)
            
# #             _f     = _make_quadratic_func(xo, yo, rng)
# #             y_c[i] = _f(x_c[i])
# #             y[i]   = _f(x[i])
            
# #             funcs.append(copy(_f))
        
# #         return x_c, y_c, x, y, funcs

# # --

# # import warnings
# # warnings.filterwarnings("ignore", category=FutureWarning)

# # from sklearn.linear_model import SGDClassifier
# # from sklearn.model_selection import train_test_split

# # def _make_sgd_func(rng, dim=5, nobs=1000, pos_mean=0, pos_std=1, neg_mean=1, neg_std=1):
# #     warnings.filterwarnings("ignore", category=FutureWarning)
# #     pos = rng.normal(pos_mean, pos_std, (nobs, dim))
# #     neg = rng.normal(neg_mean, neg_std, (nobs, dim))
# #     X   = np.vstack([pos, neg]).astype(float)
# #     X /= np.sqrt((X ** 2).sum(axis=-1, keepdims=True))
# #     y   = np.hstack([
# #         np.ones(pos.shape[0]),
# #         np.zeros(neg.shape[0])
# #     ])
    
# #     X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5)
    
# #     def _f(alphas):
# #         accs = []
# #         for alpha in alphas:
# #             sgd = SGDClassifier(learning_rate='constant', eta0=10 ** float(alpha), random_state=111)
# #             sgd = sgd.fit(X_train, y_train)
# #             preds = sgd.predict(X_test)
# #             acc = (y_test == preds).mean()
# #             accs.append(acc)
        
# #         return np.array(accs).reshape(alphas.shape)
    
# #     return _f

# # class SGDDataset:
# #     def __init__(self, x_range=[-5, 0]):
# #         self.x_range = x_range
    
# #     def sample(self, n_funcs, train_samples, test_samples, rng=None):
# #         if rng is None:
# #             rng = np.random
        
# #         x_c = rng.uniform(*self.x_range, (n_funcs, train_samples, 1))
# #         x   = rng.uniform(*self.x_range, (n_funcs, test_samples, 1))
        
# #         y_c = np.zeros((n_funcs, train_samples, 1))
# #         y   = np.zeros((n_funcs, test_samples, 1))
        
# #         funcs = []
# #         for i in range(n_funcs):
# #             _f     = _make_sgd_func(rng)
# #             y_c[i] = _f(x_c[i])
# #             y[i]   = _f(x[i])
            
# #             funcs.append(copy(_f))
        
# #         return x_c, y_c, x, y, funcs

# # # --

# # from time import time
# # from joblib import Parallel, delayed

# # class CacheDataset:
# #     def __init__(self, dataset, n_batches, n_jobs=32, **kwargs):
# #         self.n_batches = n_batches
# #         self.offset    = 0
        
# #         # self._cache = [None] * n_batches
# #         # for idx in range(n_batches):
# #         #     self._cache[idx] = dataset.sample(**kwargs)
# #         #     # Make this parallel
        
# #         jobs = [delayed(dataset.sample)(rng=np.random.RandomState(i), **kwargs) for i in range(n_batches)]
# #         self._cache = Parallel(n_jobs=n_jobs)(jobs)
        
# #     def sample(self, *args, **kwargs):
# #         out = self._cache[self.offset]
# #         self.offset = (self.offset + 1) % self.n_batches
# #         return out
