#!/usr/bin/env python

"""
    models.py
    
    Based on:
        https://arxiv.org/pdf/1807.08912.pdf
        https://github.com/StanfordASL/ALPaCA/blob/master/main/alpaca.py

    # Per Murphy "Machine Learning: A probabalistic perspective", BLR
    # 1) Should have 1 / sig_eps factor for every phi_support.transpose(1, 2)
    # 2) And should be 
    #       spread_fac = self.sig_eps + self._batch_quadform1(S, phi_query)
    # 3) And _sig_logdet should be updated
    # I think that not including this mean the priors get weighted in a non-standard way
    # Maybe that's good?  Maybe it's bad?
    # NLL is also missing a k * log(2 * pi) term, so it's not comparable
    # across dimensions
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
    def __init__(self, sig_eps, input_dim, output_dim, train_sig_eps=False):
        super().__init__()
        
        sig_eps = torch.Tensor([sig_eps])
        if train_sig_eps:
            self.sig_eps = nn.Parameter(sig_eps)
        else:
            self.register_buffer('sig_eps', sig_eps)
        
        self.register_buffer('eye', torch.eye(output_dim))
        
        self.m_prior = nn.Parameter(torch.zeros(input_dim, output_dim))
        self.S_inv_prior_asym = nn.Parameter(torch.randn(input_dim, input_dim))
        
        torch.nn.init.xavier_uniform_(self.m_prior)
        torch.nn.init.xavier_uniform_(self.S_inv_prior_asym)
        
        self.input_dim  = input_dim
        self.output_dim = output_dim
        
        self.alpha = 1
    
    def forward(self, phi_support, y_support, phi_query, y_query=None):
        S_inv_prior = self.alpha * self.S_inv_prior_asym @ self.S_inv_prior_asym.t()
        m_prior     = S_inv_prior @ self.m_prior
        
        nobs = phi_support.shape[1]
        if (nobs > 0):
            S_inv = (phi_support.transpose(1, 2) @ phi_support) + S_inv_prior[None,:]
            S     = torch.inverse(S_inv)
            m     = S @ ((phi_support.transpose(1, 2) @ y_support) + m_prior)
        else:
            S = torch.inverse(S_inv_prior[None,:]).repeat(phi_query.shape[0], 1, 1)
            m = self.m_prior[None,:]
        
        mu = phi_query @ m
        
        spread_fac = 1 + self._batch_quadform1(S, phi_query)
        sig        = torch.einsum('...i,jk->...ijk', spread_fac, self.eye * self.sig_eps)
        
        nll = None
        if y_query is not None:
            logdet = self._sig_logdet(spread_fac)
            quadf  = self._batch_quadform2(torch.inverse(sig), y_query - mu)
            nll    = logdet.mean() + quadf.mean()
        
        return mu, sig, nll
    
    def _batch_quadform1(self, A, b):
        # Eq 8 helper
        #   Also equivalent to: ((b @ A) * b).sum(dim=-1)
        
        return torch.einsum('...ij,...jk,...ik->...i', b, A, b)
    
    def _batch_quadform2(self, A, b):
        # Eq 10 helper
        return torch.einsum('...i,...ij,...j->...', b, A, b)
    
    def _sig_logdet(self, spread_fac):
        # Compute logdet(sig)
        # Equivalent to [[ss.logdet() for ss in s] for s in sig]
        return self.output_dim * (spread_fac.log() + torch.log(self.sig_eps))


# --
# NN Helper

def mixup(x_support, y_support, x_query, y_query):
    lam          = np.random.uniform(0, 1, (x_support.shape[0], 1, 1))
    pidx         = np.random.permutation(x_support.shape[0])
    x_support    = lam * x_support + (1 - lam) * x_support[pidx]
    y_support    = lam * y_support + (1 - lam) * y_support[pidx]
    x_query      = lam * x_query + (1 - lam) * x_query[pidx]
    y_query      = lam * y_query + (1 - lam) * y_query[pidx]
    return x_support, y_support, x_query, y_query


class _TrainMixin:
    def _run_loop(self, dataset, opt, batch_size=10, support_size=5, query_size=5, num_samples=100, 
        metric_fn=metrics.mean_squared_error, mixup=False):
        
        mse_hist, nll_hist = [], []
        gen = trange(num_samples // batch_size)
        for batch_idx in gen:
            if isinstance(dataset, list):
                idxs = np.random.choice(len(dataset), batch_size)
                x_support, y_support, x_query, y_query, _ =\
                    list(zip(*[dataset[idx] for idx in idxs]))
                
                x_support, y_support, x_query, y_query =\
                    [np.stack(xx) for xx in (x_support, y_support, x_query, y_query)]
                
            else:
                x_support, y_support, x_query, y_query, _ = dataset.sample_batch(
                    batch_size=batch_size,
                    support_size=support_size, # Could sample this horizon for robustness
                    query_size=query_size,     # Could sample this horizon for robustness
                )
            
            if mixup:
                x_support, y_support, x_query, y_query = mixup(x_support, y_support, x_query, y_query)
            
            inp = list2tensors((x_support, y_support, x_query, y_query), cuda=self.is_cuda)
            
            if opt is not None:
                opt.zero_grad()
                mu, sig, loss = self(*inp)
                loss.backward()
                opt.step()
            else:
                with torch.no_grad():
                    mu, sig, loss = self(*inp)
            
            nll_hist.append(float(loss))
            mse_hist.append(metric_fn(y_query, to_numpy(mu)))
            
            if not batch_idx % 10:
                gen.set_postfix(
                    mse='%0.8f' % np.mean(mse_hist[-10:]),
                    nll='%0.8f' % np.mean(nll_hist[-10:]),
                )
        
        self.nll_hist = np.array(nll_hist)
        self.mse_hist = np.array(mse_hist)
        
        return mse_hist, nll_hist
    
    def do_train(self, dataset, opt, **kwargs):
        _ = self.train()
        return self._run_loop(dataset, opt, **kwargs)
    
    def do_valid(self, dataset, opt=None, **kwargs):
        assert opt is None
        _ = self.eval()
        return self._run_loop(dataset, opt=None, **kwargs)

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


class BN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim)
    
    def forward(self, x):
        bs, obs, d = x.shape
        x = x.view(bs * obs, d)
        x = self.bn(x)
        x = x.view(bs, obs, d)
        return x


class Block(nn.Module):
    def __init__(self, input_dim, output_dim, act, bn=True, skip=False):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.act    = act()
        self.bn     = BN(output_dim) if bn else None
        self.skip   = skip
    
    def forward(self, x_inp):
        x = self.linear(x_inp)
        
        if self.bn:
            x = self.bn(x)
        
        x = self.act(x)
        
        if self.skip:
            x = x + x_inp
        
        return x


class ALPACA(_TrainMixin, nn.Module):
    def __init__(self, input_dim, output_dim, sig_eps, num=1, activation='Tanh', 
        hidden_dim=128, train_sig_eps=False):
        
        super().__init__()
        
        _act = getattr(nn, activation)
        
        self.backbone = nn.Sequential(
            Block(input_dim, hidden_dim, _act, bn=True),
            Block(hidden_dim, hidden_dim, _act, bn=True, skip=False),
            Block(hidden_dim, hidden_dim, _act, bn=True, skip=False),
        )
        
        self.blr = BLR(sig_eps=sig_eps, input_dim=hidden_dim, 
            output_dim=output_dim, train_sig_eps=train_sig_eps)
        
        self.num        = num
        self.input_dim  = input_dim
        self.output_dim = output_dim
    
    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda
    
    def forward(self, x_support, y_support, x_query, y_query=None):
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



# --
# Alternate BLR

# class BLR(nn.Module):
#     def __init__(self, sig_eps, input_dim, output_dim, train_sig_eps=False):
#         super().__init__()
        
#         sig_eps = 10 ** torch.Tensor([sig_eps])
#         if train_sig_eps:
#             self.sig_eps = nn.Parameter(sig_eps)
#         else:
#             self.register_buffer('sig_eps', sig_eps)
        
#         self.register_buffer('eye', torch.eye(output_dim))
        
#         self.m_prior = nn.Parameter(torch.zeros(input_dim, output_dim))
#         self.S_inv_prior_asym = nn.Parameter(torch.randn(input_dim, input_dim))
        
#         torch.nn.init.xavier_uniform_(self.m_prior)
#         torch.nn.init.xavier_uniform_(self.S_inv_prior_asym)
        
#         self.input_dim  = input_dim
#         self.output_dim = output_dim
        
#         self.alpha = 1
    
#     def forward(self, phi_support, y_support, phi_query, y_query=None):
#         # !! Control s_inv and m_prior weight separately?
#         S_inv_prior = self.alpha * self.S_inv_prior_asym @ self.S_inv_prior_asym.t()
#         m_prior     = S_inv_prior @ self.m_prior
        
#         nobs = phi_support.shape[1]
#         if (nobs > 0):
#             S_inv = (1 / self.sig_eps * phi_support.transpose(1, 2) @ phi_support) + S_inv_prior[None,:]
#             S     = torch.inverse(S_inv)
#             m     = S @ (1 / self.sig_eps * phi_support.transpose(1, 2) @ y_support + m_prior)
#         else:
#             S = torch.inverse(S_inv_prior[None,:]).repeat(phi_query.shape[0], 1, 1)
#             m = self.m_prior[None,:]
        
#         mu = phi_query @ m
        
#         spread_fac = self.sig_eps + self._batch_quadform1(S, phi_query)
#         sig        = torch.einsum('...i,jk->...ijk', spread_fac, self.eye)
        
#         predictive_nll = None
#         if y_query is not None:
#             quadf = self._batch_quadform2(torch.inverse(sig), y_query - mu)
#             predictive_nll = self._sig_logdet(spread_fac).mean() + quadf.mean()
        
#         return mu, sig, predictive_nll
    
#     def _batch_quadform1(self, A, b):
#         # Eq 8 helper
#         #   Also equivalent to: ((b @ A) * b).sum(dim=-1)
#         return torch.einsum('...ij,...jk,...ik->...i', b, A, b)
    
#     def _batch_quadform2(self, A, b):
#         # Eq 10 helper
#         return torch.einsum('...i,...ij,...j->...', b, A, b)
    
#     def _sig_logdet(self, spread_fac):
#         # Compute logdet(sig)
#         # Equivalent to [[ss.logdet() for ss in s] for s in sig]
#         return self.output_dim * spread_fac.log()
