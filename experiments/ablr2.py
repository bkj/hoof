#!/usr/bin/env python

"""
    ablr2.py
"""

from rsub import *
from matplotlib import pyplot as plt

import numpy as np

import torch
from torch import nn

from hoof import dataset
from hoof.helpers import set_seeds, to_numpy
from hoof.helpers import HoofMetrics as metrics

torch.set_default_tensor_type('torch.DoubleTensor')
torch.set_num_threads(1)
set_seeds(123)

from hoof.models import ALPACA
from hoof.models import BLR as OBLR

# --
# Helpers

class Encoder(nn.Module):
    """ NN for learning projection """
    def __init__(self, input_dim=1, output_dim=1, hidden_dim=64):
        super().__init__()
        self._encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
    
    def forward(self, x):
        return self._encoder(x)



def compute_blr_nll(phi, y, alpha, beta):
    n       = phi.shape[0]
    sigma_t = (1 / alpha) * (phi @ phi.t()) + (1 / beta) * torch.eye(n)
    nll     = 0.5 * (sigma_t.logdet() + y.t() @ torch.inverse(sigma_t) @ y).squeeze()
    return nll / n


class BLR:
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta  = beta
        
    def fit(self, phi, y):
        n, d = phi.shape
        
        S_inv_prior = self.alpha * torch.eye(d)
        S_inv       = S_inv_prior + self.beta * (phi.t() @ phi)
        S           = torch.inverse(S_inv)
        
        m  = self.beta * S @ phi.t() @ y
        
        self.m = m
        self.S = S
        
        return self
    
    def score_predict(self, phi, y):
        mu  = phi @ self.m
        sig = (1 / self.beta) + ((phi @ self.S) * phi).sum(dim=-1, keepdim=True)
        
        resid_z = ((y - mu) ** 2) / sig
        nll     = (sig.log() + resid_z).mean()
        
        mse = metrics.mean_squared_error(y, mu)
        
        return mu, sig, nll, mse



# --
# Make dataset

num_problems = 30

dataset_name  = 'SinusoidDataset'
dataset_cls   = getattr(dataset, dataset_name)
train_dataset = dataset_cls()
valid_dataset = dataset_cls()

def make_dataset(dataset, num_problems):
    problems, fns = [], []
    for _ in range(num_problems):
        x_s, y_s, x_q, y_q, fn = dataset.sample_one(support_size=5, query_size=5)
        problems.append([
            torch.Tensor(x_s),
            torch.Tensor(y_s) + (torch.randn(y_s.shape) * 0.1),
            torch.Tensor(x_q),
            torch.Tensor(y_q) + (torch.randn(y_q.shape) * 0.1),
        ])
        fns.append(fn)
    
    return problems, fns

train_problems, train_fns = make_dataset(train_dataset, num_problems)
valid_problems, valid_fns = make_dataset(valid_dataset, 10 * num_problems)

# >>
# ALPACA

model = ALPACA(input_dim=1, output_dim=1, sig_eps=0.01, hidden_dim=64, activation='tanh').cuda()

train_history = []
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

train_kwargs = {"batch_size" : 64, "support_size" : 10, "query_size" : 100, "num_samples" : 30000}

train_history += model.train(dataset=train_problems, opt=opt, **train_kwargs)

np.mean(model.valid(dataset=valid_problems))

_ = plt.plot(train_history, c='red', label='train')
_ = plt.yscale('log')
_ = plt.grid()
_ = plt.legend()
show_plot()

# <<

# --
# Setup model

encoder     = Encoder()
log_alphas  = nn.Parameter(0 + torch.zeros(num_problems)) # prior weight
log_betas   = nn.Parameter(1 + torch.zeros(num_problems)) # noise variance

params = list(encoder.parameters()) + [log_alphas, log_betas]
opt    = torch.optim.LBFGS(params, max_iter=100, tolerance_change=1e-4)

# --
# Train

def _target_fn():
    opt.zero_grad()
    
    total_nll, total_mse = 0, 0
    for idx, (X, y) in enumerate(train_problems):
        alpha, beta = 10 ** log_alphas[idx], 1 / 10 ** log_betas[idx]
        
        phi = encoder(X)
        
        blr = BLR(alpha=alpha, beta=beta)
        blr = blr.fit(phi, y)
        mu, sig, nll, mse = blr.score_predict(phi, y)
        
        total_nll += nll
        total_mse += metrics.mean_squared_error(y.squeeze(), mu.squeeze())
    
    total_nll /= len(train_problems)
    total_nll.backward()
    nn.utils.clip_grad_norm_(params, max_norm=1)
    
    total_mse /= len(train_problems)
    
    print(float(total_mse), float(total_nll))
    
    return float(total_nll)


_ = opt.step(_target_fn)


# --

# class OBLR(nn.Module):
#     # !! Seems to be most stable version
#     def __init__(self, alpha, sig_eps, input_dim, output_dim):
#         super().__init__()
        
#         self.alpha   = alpha
#         self.sig_eps = sig_eps
        
#         self.register_buffer('inp_eye', torch.eye(input_dim))
#         self.register_buffer('out_eye', torch.eye(output_dim))
        
#         self.m_prior   = nn.Parameter(torch.zeros(input_dim, output_dim))
#         self.S_inv_fac = nn.Parameter(torch.randn(input_dim, input_dim))
        
#         torch.nn.init.xavier_uniform_(self.m_prior)
#         torch.nn.init.xavier_uniform_(self.S_inv_fac)
        
#         self.input_dim  = input_dim
#         self.output_dim = output_dim
    
#     def forward(self, phi, y, phi_q, y_q):
        
#         try:
            
#             S_inv_prior = self.alpha * (self.S_inv_fac @ self.S_inv_fac.t())
            
#             S_inv   = S_inv_prior + (phi.t() @ phi)
#             S       = torch.inverse(S_inv)
#             m       = S @ (phi.t() @ y + S_inv_prior @ self.m_prior)
            
#             mu  = phi_q @ m
            
#             sig = self.sig_eps * (1 + ((phi_q @ S) * phi_q).sum(dim=-1, keepdim=True))
#             sig = sig.clamp(min=1e-6)
            
#             nll = ((y_q - mu).pow(2) / sig).mean() + sig.log().mean()
            
#             if torch.isnan(nll):
#                 print('nll null')
#                 raise Exception
#         except:
#             print('phi', phi)
#             print('(phi.t() @ phi)', (phi.t() @ phi).squeeze())
#             print('S_inv', S_inv.squeeze())
#             print('S', S.squeeze())
#             print('m', m.squeeze())
#             print('mu', mu.squeeze())
#             print('sig', sig.squeeze())
#             raise Exception
        
#         return mu, sig, nll


encoder     = Encoder()
log_alpha   = nn.Parameter(0 + torch.zeros(num_problems)) # prior weight
log_sig_eps = nn.Parameter(-2 + torch.zeros(num_problems)) # prior weight
params      = list(encoder.parameters()) + [log_alpha, log_sig_eps]
opt         = torch.optim.LBFGS(params, max_iter=100, tolerance_change=1e-5)

def _target_fn():
    opt.zero_grad()
    
    total_nll, total_mse = 0, 0
    for idx, (X, y, X_q, y_q) in enumerate(train_problems):
        phi   = encoder(X)
        phi_q = encoder(X_q)
        
        # alpha   = 10 ** log_alpha[idx]
        # sig_eps = 10 ** log_sig_eps[idx]
        
        blr          = OBLR(sig_eps=0.1, input_dim=phi.shape[1], output_dim=1)
        mu, sig, nll = blr(phi[None,...], y[None,...], phi_q[None,...], y_q[None,...])
        mu, sig      = mu.squeeze(dim=0), sig.squeeze(dim=0)
        
        total_nll += nll
        total_mse += metrics.mean_squared_error(y_q.squeeze(), mu.squeeze())
    
    total_nll /= len(train_problems)
    total_nll.backward()
    
    total_mse /= len(train_problems)
    
    print(float(total_mse), float(total_nll))
    
    return total_nll

_ = opt.step(_target_fn)


