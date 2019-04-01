#!/usr/bin/env python

"""
    simple_example.py
"""

import sys
import numpy as np
from time import time
from tqdm import tqdm, trange

import torch
from torch import nn

from rsub import *
from matplotlib import pyplot as plt

from hoof import dataset
from hoof.models import ALPACA, rks_regression
from hoof.helpers import set_seeds, to_numpy, list2tensors, tensors2list, set_lr

from sklearn.linear_model import LinearRegression

torch.set_num_threads(1)
set_seeds(345)

# --
# Define model

train_dataset = dataset.SinusoidDataset()

model = ALPACA(input_dim=1, output_dim=1, hidden_dim=64, sig_eps=0.1, activation='tanh').cuda()
opt   = torch.optim.Adam(model.parameters())

train_history = []
for support_size in list(range(2, 11))[::-1]:
    train_history += model.train(dataset=train_dataset, opt=opt, support_size=support_size, query_size=100, num_samples=10000)

_ = plt.plot(train_history)
_ = plt.yscale('log')
_ = plt.grid()
show_plot()


# model.valid(dataset=train_dataset, support_size=10, query_size=100, num_samples=10000)

# Plot samples
for _ in range(10):
    x_s, y_s, _, _, _ = train_dataset.sample_one(support_size=100, query_size=0)
    _ = plt.scatter(x_s.squeeze(), y_s.squeeze())

show_plot()

# Embed line
X_grid = torch.linspace(-5, 5, 256).view(-1, 1)
X_grid[X_grid.abs().argmin()] = 0
emb    = to_numpy(model.backbone(X_grid.cuda()))

# Plot basis functions
for i in range(emb.shape[1]):
    _ = plt.plot(scale_emb[:,i], alpha=0.25)

show_plot()

scale_emb = emb.copy()
scale_emb -= scale_emb.mean(axis=0)
scale_emb /= scale_emb.std(axis=0)

# Plot scaled basis functions
for i in range(emb.shape[1]):
    _ = plt.plot(scale_emb[:,i], alpha=0.25)

show_plot()

pca  = PCA(n_components=10)
pemb = pca.fit_transform(scale_emb)
_ = plt.plot(pca.explained_variance_ratio_)
show_plot()

for i in range(10):
    _ = plt.plot(pemb[:,i])

show_plot()


fn = train_dataset.sample_fn()

sel = np.random.choice(X_grid.shape[0], 2)
xx  = X_grid[sel]
yy  = fn(xx)

phi  = emb[sel]
sphi = scale_emb[sel]
pphi = pemb[sel]

mu, sig, _ = model(xx[:i].cuda(), yy[:i].cuda(), X_grid.cuda())
mu  = to_numpy(mu.squeeze())
sig = to_numpy(sig.squeeze())

(X_grid.squeeze() == 0).nonzero()
mu[sel]
mu[127]


_ = plt.plot(to_numpy(X_grid), mu + np.random.normal(0, 0.1, pred.shape), c='green', alpha=0.75)
_ = plt.fill_between(to_numpy(X_grid).squeeze(), mu - 1.96 * np.sqrt(sig), mu + 1.96 * np.sqrt(sig), alpha=0.2)

pred = LinearRegression().fit(phi, yy).predict(emb).squeeze()
_ = plt.plot(to_numpy(X_grid), pred + np.random.normal(0, 0.1, pred.shape), c='red', alpha=0.75) # No prior

# pred = LinearRegression().fit(sphi, yy).predict(scale_emb).squeeze()
# _ = plt.plot(to_numpy(X_grid), pred + np.random.normal(0, 0.1, pred.shape), c='green')

# pred = LinearRegression().fit(pphi, yy).predict(pemb).squeeze()
# _ = plt.plot(to_numpy(X_grid), pred + np.random.normal(0, 0.1, pred.shape), c='blue')

_ = plt.plot(to_numpy(X_grid), to_numpy(fn(X_grid)), c='black', alpha=0.75)
_ = plt.scatter(xx, yy, s=100, c='black')
show_plot()

prior = to_numpy(torch.Tensor(emb) @ model.blr.K.cpu()).squeeze()
_ = plt.plot(prior)
show_plot()

model_outputs = [model(xx[:i].cuda(), yy[:i].cuda(), X_grid.cuda()) for i in range(xx.shape[0])]

mus = [to_numpy(m[0]).squeeze() for m in model_outputs]
_ = [plt.plot(to_numpy(X_grid), mu, label=i) for i, mu in enumerate(mus)]
_ = plt.plot(to_numpy(X_grid), to_numpy(fn(X_grid)), c='black', alpha=0.75)
_ = plt.scatter(xx, yy, s=100, c='black')
_ = plt.legend()
show_plot()


torch.cat([
    torch.ones(5, 10, 1),
    torch.randn(5, 10, 4)
], dim=-1)

# --

import torch
from torch import nn

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
        self.L_asym = nn.Parameter(torch.zeros(input_dim, input_dim))
        
        # torch.nn.init.xavier_uniform_(self.K)
        # torch.nn.init.xavier_uniform_(self.L_asym)
        
        self.input_dim  = input_dim
        self.output_dim = output_dim
    
    def forward(self, phi_support, y_support, phi_query, y_query=None):
        L = self.L_asym @ self.L_asym.t()
        
        nobs = phi_support.shape[1]
        if (nobs > 0):
            posterior_L     = (phi_support.transpose(1, 2) @ phi_support) + L[None,:]
            posterior_L_inv = torch.inverse(posterior_L)
            posterior_K     = posterior_L_inv @ ((phi_support.transpose(1, 2) @ y_support) + (L @ self.K))
        else:
            posterior_L_inv = torch.inverse(L[None,:]).repeat(phi_query.shape[0], 1, 1)
            posterior_K     = self.K[None,:]
        
        mu_pred = phi_query @ posterior_K
        
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

# from hoof.models import BLR

blr = BLR(sig_eps=0.01, input_dim=2, output_dim=1)

x = np.random.uniform(0, 1, (2, 1))
y = np.random.uniform(0, 1, (2, 1))

x = np.column_stack([np.ones(x.shape[0]), x])

mu, sig, _ = blr.forward(torch.Tensor(x)[None,...], torch.Tensor(y)[None,...], torch.Tensor(x)[None,...])

mu.squeeze()
y.squeeze()












