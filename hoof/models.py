#!/usr/bin/env python

"""
    models.py
    
    Based on:
        https://arxiv.org/pdf/1807.08912.pdf
        https://github.com/StanfordASL/ALPaCA/blob/master/main/alpaca.py
"""

import sys
import numpy as np
from tqdm import trange

from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LinearRegression

import torch
from torch import nn

from hoof.helpers import set_seeds, to_numpy, list2tensors, tensors2list
from hoof.helpers import HoofMetrics as metrics

# --
# Bayesian Linear Regression

class BLR(nn.Module):
    def __init__(self, sig_eps, input_dim, output_dim):
        super().__init__()
        
        # !! Should sig_eps be traininable?
        # !! Notices that not training K or L_asym doesn't make a big difference
        # !! Is the bias a good idea?
        
        self.sig_eps     = sig_eps
        self.log_sig_eps = np.log(sig_eps)
        self.register_buffer('sig_eps_eye', torch.eye(output_dim) * self.sig_eps)
        
        self.K      = nn.Parameter(torch.zeros(input_dim, output_dim))
        self.L_asym = nn.Parameter(torch.randn(input_dim, input_dim))
        self.bias   = nn.Parameter(torch.zeros([1]))
        
        torch.nn.init.xavier_uniform_(self.K)
        torch.nn.init.xavier_uniform_(self.L_asym)
        
        self.input_dim  = input_dim
        self.output_dim = output_dim
    
    def forward(self, phi_support, y_support, phi_query, y_query=None):
        L = self.L_asym @ self.L_asym.t()
        
        nobs = phi_support.shape[1]
        if nobs > 0:
            posterior_L     = (phi_support.transpose(1, 2) @ phi_support) + L[None,:]
            posterior_L_inv = torch.inverse(posterior_L)
            posterior_K     = posterior_L_inv @ ((phi_support.transpose(1, 2) @ y_support) + (L @ self.K))
        else:
            posterior_L_inv = torch.inverse(L)
            posterior_K     = self.K # !! According to original code
        
        mu_pred = phi_query @ posterior_K + self.bias
        
        spread_fac = 1 + self._batch_quadform1(posterior_L_inv, phi_query)
        
        # Expand each element of spread_fac to y_dim diagonal matrix
        sig_pred = torch.einsum('...i,jk->...ijk', spread_fac, self.sig_eps_eye)
        
        predictive_nll = None
        if y_query is not None:
            quadf = self._batch_quadform2(torch.inverse(sig_pred), y_query - mu_pred)
            predictive_nll = self._sig_pred_logdet(spread_fac).mean() + quadf.mean()
        
        return mu_pred, sig_pred, predictive_nll
    
    def _batch_quadform1(self, A, b):
        # Eq 8 helper
        #   Also equivalent to: ((b @ A) * b).sum(dim=-1)
        return torch.einsum('...ij,...jk,...ik->...i', b, A, b)
    
    def _batch_quadform2(self, A, b):
        # Eq 10 helper
        return torch.einsum('...i,...ij,...j->...', b, A, b)
    
    def _sig_pred_logdet(self, spread_fac):
        # Compute logdet(sig_pred)
        # Equivalent to [[ss.logdet() for ss in s] for s in sig_pred]
        return self.output_dim * (spread_fac.log() + self.log_sig_eps)

# --
# NN Helper

class _TrainMixin:
    def _run_epoch(self, dataset, opt, batch_size=10, support_size=5, query_size=5, num_samples=100, metric_fn=metrics.mean_squared_error):
        hist = []
        gen = trange(num_samples // batch_size)
        for batch_idx in gen:
            x_support, y_support, x_query, y_query, _ = dataset.sample_batch(
                batch_size=batch_size,
                support_size=support_size, # Could sample this horizon for robustness
                query_size=query_size,     # Could sample this horizon for robustness
            )
            
            inp = list2tensors((x_support, y_support, x_query, y_query), cuda=self.is_cuda)
            
            if opt is not None:
                opt.zero_grad()
                mu, sig, loss = self(*inp)
                loss.backward()
                opt.step()
            else:
                with torch.no_grad():
                    mu, sig, loss = self(*inp)
            
            hist.append(metric_fn(y_query, to_numpy(mu)))
            
            if not batch_idx % 10:
                gen.set_postfix(loss='%0.8f' % np.mean(hist[-10:]))
            
        return hist
    
    def train(self, dataset, opt, **kwargs):
        return self._run_epoch(dataset, opt, **kwargs)
    
    def valid(self, dataset, opt=None, **kwargs):
        assert opt is None
        return self._run_epoch(dataset, opt=None, **kwargs)

# --
# ALPACA

def _check_shapes(x_support, y_support, x_query, y_query, input_dim, output_dim):
    assert x_support.shape[-1] == input_dim
    assert y_support.shape[-1] == output_dim
    assert x_query.shape[-1] == input_dim
    if y_query is not None:
        assert y_query.shape[-1] == output_dim
    
    if len(x_support.shape) == 2: x_support = x_support[None,:]
    if len(y_support.shape) == 2: y_support = y_support[None,:]
    if len(x_query.shape) == 2:   x_query = x_query[None,:]
    
    return x_support, y_support, x_query, y_query

class ALPACA(_TrainMixin, nn.Module):
    def __init__(self, input_dim, output_dim, sig_eps, num=1, activation='tanh', hidden_dim=128):
        super().__init__()
        
        if activation == 'tanh':
            _act = nn.Tanh
        elif activation  == 'relu':
            _act = nn.ReLU
        else:
            raise Exception('!! unknown activation %s' % activation)
        
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, hidden_dim),
            _act() # Do we want this?
        )
        
        self.blr = BLR(sig_eps=sig_eps, input_dim=hidden_dim, output_dim=output_dim)
        
        self.num        = num
        self.input_dim  = input_dim
        self.output_dim = output_dim
    
    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda
    
    def forward(self, x_support, y_support, x_query, y_query=None):
        # !! POTENTIAL BUG: should be normalizing x_support, y_support, x, and y
        
        x_support, y_support, x_query, y_query =\
            _check_shapes(x_support, y_support, x_query, y_query, self.input_dim, self.output_dim)
        
        phi_support = self.backbone(x_support)
        phi_query   = self.backbone(x_query)
        
        return self.blr(phi_support, y_support, phi_query, y_query)

# --
# Random Kitchen Sinks

def rks_regression(x_s, y_s, x_grid, **kwargs):
    rbf = RBFSampler(**kwargs)
    p_s = rbf.fit_transform(x_s)
    lr  = LinearRegression().fit(p_s, y_s)
    return lr.predict(rbf.transform(x_grid))
