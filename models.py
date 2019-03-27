#!/usr/bin/env python

"""
    models.py
"""

import torch
from torch import nn

class ALPACA(nn.Module):
    # !! ignoring `f_nom` from the original
    def __init__(self, x_dim, y_dim, sig_eps, num=1, activation='tanh'):
        super().__init__()
        
        final_dim = 128
        
        self.sig_eps  = sig_eps
        self.eye      = torch.eye(y_dim)
        
        # Seems to work OK even if this is not trainable
        self.K      = nn.Parameter(torch.zeros(final_dim, y_dim)) # last layer size
        self.L_asym = nn.Parameter(torch.randn(final_dim, final_dim))    # last layer size
        
        torch.nn.init.xavier_uniform_(self.K)
        torch.nn.init.xavier_uniform_(self.L_asym)
        
        if activation == 'tanh':
            act = nn.Tanh
        elif activation  == 'relu':
            act = nn.ReLU
        else:
            raise Exception('!! unknown activation %s' % activation)
        
        self.backbone = nn.Sequential(
            nn.Linear(x_dim, 128),
            act(),
            nn.Linear(128, 128),
            act(),
            nn.Linear(128, final_dim),
            act() # Do we want this?
        )
        
        self.num   = num
        self.x_dim = x_dim
        self.y_dim = y_dim
    
    def forward(self, x_c, y_c, x, y=None):
        assert x_c.shape[-1] == self.x_dim
        assert y_c.shape[-1] == self.y_dim
        assert x.shape[-1] == self.x_dim
        if y is not None:
            assert y.shape[-1] == self.y_dim
        
        L = torch.mm(self.L_asym, self.L_asym.t())
        
        phi = self.backbone(x)
        
        nobs = x_c.shape[1]
        if nobs > 0:
            phi_c = self.backbone(x_c)
            
            posterior_L_inv = torch.bmm(phi_c.transpose(1, 2), phi_c)
            posterior_L_inv = posterior_L_inv + L.unsqueeze(0)
            posterior_L_inv = self.batch_inv(posterior_L_inv)
            
            posterior_K = torch.bmm(phi_c.transpose(1, 2), y_c) + torch.mm(L, self.K)
            posterior_K = torch.bmm(posterior_L_inv, posterior_K)
        else:
            # nobs == 0 means we have no observations, so this is modeling the prior
            # skip for now
            raise Exception()
            posterior_K, posterior_L_inv = self.K, torch.inverse(L)
            posterior_K     = posterior_K.unsqueeze(0).repeat(30, 1, 1)
            posterior_L_inv = posterior_L_inv.unsqueeze(0).repeat(30, 1, 1)
        
        mu_pred = torch.bmm(phi, posterior_K)
        
        spread_fac = 1 + self.batch_quadform(posterior_L_inv, phi)
        sig_pred = spread_fac.unsqueeze(-1) * self.eye.view(1, 1, self.y_dim, self.y_dim) * self.sig_eps
        
        predictive_nll = None
        if y is not None:
            logdet = self.y_dim * spread_fac.log() + (self.eye * self.sig_eps).logdet()
            quadf  = self.batch_quadform2(self.batch_inv(sig_pred), y - mu_pred)
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
