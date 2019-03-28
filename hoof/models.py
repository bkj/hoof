#!/usr/bin/env python

"""
    models.py
    
    Based on:
        https://arxiv.org/pdf/1807.08912.pdf
        https://github.com/StanfordASL/ALPaCA/blob/master/main/alpaca.py
"""

import sys
import numpy as np

import torch
from torch import nn

class ALPACA(nn.Module):
    # !! ignoring `f_nom` from the original
    def __init__(self, x_dim, y_dim, sig_eps, num=1, activation='tanh', hidden_dim=128, final_dim=128):
        super().__init__()
        
        self.sig_eps  = nn.Parameter(torch.Tensor([sig_eps])) # BKJ: made this trainable
        self.register_buffer('eye', torch.eye(y_dim))
        
        # Seems to work OK even if these are not trainable (these are just the priors, so makes sense)
        self.K      = nn.Parameter(torch.zeros(final_dim, y_dim))
        self.L_asym = nn.Parameter(torch.randn(final_dim, final_dim))
        self.bias   = nn.Parameter(torch.zeros([1])) # BKJ: added this
        
        torch.nn.init.xavier_uniform_(self.K)
        torch.nn.init.xavier_uniform_(self.L_asym)
        
        if activation == 'tanh':
            _act = nn.Tanh
        elif activation  == 'relu':
            _act = nn.ReLU
        else:
            raise Exception('!! unknown activation %s' % activation)
        
        self.backbone = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, final_dim),
            _act() # Do we want this?
        )
        
        self.num   = num
        self.x_dim = x_dim
        self.y_dim = y_dim
    
    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda
    
    def forward(self, x_support, y_support, x_query, y_query=None):
        # !! BUG: should be normalizing x_support, y_support, x, and y
        assert x_support.shape[-1] == self.x_dim
        assert y_support.shape[-1] == self.y_dim
        assert x_query.shape[-1] == self.x_dim
        if y_query is not None:
            assert y_query.shape[-1] == self.y_dim
        
        if len(x_support.shape) == 2: x_support = x_support[None,:]
        if len(y_support.shape) == 2: y_support = y_support[None,:]
        if len(x_query.shape) == 2:   x_query = x_query[None,:]
        
        L = torch.mm(self.L_asym, self.L_asym.t())
        
        nobs = x_support.shape[1]
        if nobs > 0:
            phi_support = self.backbone(x_support)
            
            posterior_L_inv = torch.bmm(phi_support.transpose(1, 2), phi_support)
            posterior_L_inv = posterior_L_inv + L.unsqueeze(0)
            posterior_L_inv = self.batch_inv(posterior_L_inv)
            
            posterior_K = torch.bmm(phi_support.transpose(1, 2), y_support) + torch.mm(L, self.K)
            posterior_K = torch.bmm(posterior_L_inv, posterior_K)
        else:
            posterior_L_inv = self.batch_inv(L)
            posterior_K     = self.K
        
        phi     = self.backbone(x_query)
        mu_pred = torch.bmm(phi, posterior_K) + self.bias
        
        spread_fac = 1 + self.batch_quadform(posterior_L_inv, phi)
        sig_pred = spread_fac.unsqueeze(-1) * self.eye.view(1, 1, self.y_dim, self.y_dim) * self.sig_eps
        
        predictive_nll = None
        if y_query is not None:
            logdet = self.y_dim * spread_fac.log() + (self.eye * self.sig_eps).logdet()
            quadf  = self.batch_quadform2(self.batch_inv(sig_pred), y_query - mu_pred)
            predictive_nll = (logdet + quadf).mean()
        
        return mu_pred, sig_pred, predictive_nll
    
    def batch_inv(self, x):
        eye = x.new_ones(x.size(-1)).diag().expand_as(x)
        x_inv, _ = torch.gesv(eye, x)
        return x_inv
    
    def batch_quadform(self, A, b):
        bs, hor, hdim = b.shape
        
        b_vec = b.unsqueeze(-1)
        Ab = torch.bmm(A, b.transpose(1, 2)).transpose(1, 2).unsqueeze(-1)
        
        b_vec = b_vec.reshape(bs * hor, hdim, 1)
        Ab    = Ab.reshape(bs * hor, hdim, 1)
        out   = torch.bmm(b_vec.transpose(1, 2), Ab).squeeze(-1)
        out   = out.reshape(bs, hor, 1)
        return out
    
    def batch_quadform2(self, A, b):
        bs, hor = A.shape[0], A.shape[1]
        
        b_vec   = b.unsqueeze(-1)
        b_vec_t = b_vec.transpose(2, 3)
        
        A       = A.reshape(A.shape[0] * A.shape[1], A.shape[2], A.shape[3])
        b_vec   = b_vec.reshape(b_vec.shape[0] * b_vec.shape[1], b_vec.shape[2], b_vec.shape[3])
        b_vec_t = b_vec_t.reshape(b_vec_t.shape[0] * b_vec_t.shape[1], b_vec_t.shape[2], b_vec_t.shape[3])
        
        out = torch.bmm(b_vec_t, A)
        out = torch.bmm(out, b_vec)
        out = out.reshape(bs, hor, 1)
        return out

# # --
# # Simplified version of the above
# # Has not been tested to guarantee same results as original version

# class ALPACA2(nn.Module):
#     # !! ignoring `f_nom` from the original
#     def __init__(self, x_dim, y_dim, sig_eps, activation='Tanh', hidden_dim=128, final_dim=128):
#         super().__init__()
        
#         self.sig_eps     = sig_eps
#         self.log_sig_eps = np.log(sig_eps)
#         self.sig_eps_eye = torch.eye(self.y_dim) * self.sig_eps
        
#         # Seems to work OK even if this is not trainable
#         self.K      = nn.Parameter(torch.zeros(final_dim, y_dim))     # last layer size
#         self.L_asym = nn.Parameter(torch.randn(final_dim, final_dim)) # last layer size
        
#         torch.nn.init.xavier_uniform_(self.K)
#         torch.nn.init.xavier_uniform_(self.L_asym)
        
#         _act = getattr(nn, activation)
#         self.backbone = nn.Sequential(
#             nn.Linear(x_dim, hidden_dim),
#             _act(),
#             nn.Linear(hidden_dim, hidden_dim),
#             _act(),
#             nn.Linear(hidden_dim, final_dim),
#             _act(), # Do we want this?
#         )
        
#         self.x_dim = x_dim
#         self.y_dim = y_dim
    
#     def forward(self, x_support, y_support, x, y=None):
#         assert x_support.shape[-1] == self.x_dim
#         assert y_support.shape[-1] == self.y_dim
#         assert x.shape[-1] == self.x_dim
#         if y is not None:
#             assert y.shape[-1] == self.y_dim
        
#         L = self.L_asym @ self.L_asym.t()
        
#         nobs = x_support.shape[1]
#         if nobs > 0:
#             phi_support = self.backbone(x_support)
            
#             posterior_L_inv = torch.inverse((phi_support.transpose(1, 2) @ phi_support) + L[None,:])
#             posterior_K     = posterior_L_inv @ ((phi_support.transpose(1, 2) @ y_support) + (L @ self.K))
#         else:
#             posterior_L_inv = torch.inverse(L)
#             posterior_K     = self.K # !! According to original code
            
        
#         phi = self.backbone(x)
#         mu_pred = phi @ posterior_K
        
#         spread_fac = 1 + self._batch_quadform1(posterior_L_inv, phi)
        
#         # Expand each element of spread_fac to y_dim diagonal matrix
#         sig_pred = torch.einsum('...i,jk->...ijk', spread_fac, self.sig_eps_eye)
        
#         predictive_nll = None
#         if y is not None:
#             quadf = self._batch_quadform2(torch.inverse(sig_pred), y - mu_pred)
#             predictive_nll = self._sig_pred_logdet(spread_fac).mean() + quadf.mean()
        
#         return mu_pred, sig_pred, predictive_nll
    
#     def _batch_quadform1(self, A, b):
#         # Eq 8 helper
#         #   Also equivalent to: ((b @ A) * b).sum(dim=-1)
#         return torch.einsum('...ij,...jk,...ik->...i', b, A, b)
    
#     def _batch_quadform2(self, A, b):
#         # Eq 10 helper
#         return torch.einsum('...i,...ij,...j->...', b, A, b)
    
#     def _sig_pred_logdet(self, spread_fac):
#         # Compute logdet(sig_pred)
#         # Equivalent to [[ss.logdet() for ss in s] for s in sig_pred]
#         return self.y_dim * (spread_fac.log() + self.log_sig_eps)

