from rsub import *
from matplotlib import pyplot as plt

import numpy as np

import torch
from torch import nn

from hoof import dataset
from hoof.helpers import set_seeds, to_numpy

torch.set_default_tensor_type('torch.DoubleTensor')
torch.set_num_threads(1)

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


# def compute_blr_nll(phi, y, alpha, beta):
#     """
#         function for computing BLR negative log-liklihood
#         follows Eq(1) in supplementary material
#     """
#     sigma_t = (1 / alpha) * (phi @ phi.t()) + (1 / beta) * torch.eye(phi.shape[0])
#     return 0.5 * (sigma_t.logdet() + y.t() @ torch.inverse(sigma_t) @ y).squeeze()

def compute_blr_nll(phi, y, alpha, beta):
    n = phi.shape[0]
    d = phi.shape[1]
    
    s_inv = alpha * torch.eye(d) + beta * (phi.t() @ phi)
    s = torch.inverse(s_inv)
    m = beta * s @ phi.t() @ y
    
    mll = d / 2 * alpha.log()
    mll += n / 2 * beta.log()
    mll -= n / 2 * np.log(2 * np.pi)
    mll -= beta / 2 * (y - (phi @ m)).pow(2).sum().sqrt()
    mll -= alpha / 2 * (m.t() @ m).squeeze()
    mll -= 0.5 * s_inv.logdet()
    
    return - mll


class BLR:
    """ Bayesian linear regression """
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta  = beta
    
    def fit(self, phi, y):
        """ follows equation 3.53 and 3.54 in Bishop 2006 (PRML) """
        d     = phi.shape[1]
        s_inv = self.alpha * torch.eye(d) + self.beta * (phi.t() @ phi)
        
        self.s = torch.inverse(s_inv)             # posterior_L_inv
        self.m = self.beta * self.s @ phi.t() @ y # posterior_K
        return self
    
    def predict(self, phi):
        """ follows equation 3.58 and 3.59 in Bishop 2006 (PRML) """
        mu    = phi @ self.m
        sig   = 1 / self.beta + ((phi @ self.s) * phi).sum(dim=-1)
        return mu, sig
    
    def score(self, phi, y):
        self.fit(phi, y)
        mu, sig = self.predict(phi)
        mse = float((y - mu).pow(2).mean())
        return mse

# --
# Make dataset

num_problems = 10

dataset_name  = 'SinusoidDataset'
dataset_cls   = getattr(dataset, dataset_name)
train_dataset = dataset_cls()
valid_dataset = dataset_cls()

def make_dataset(dataset, num_problems):
    problems, fns = [], []
    for _ in range(num_problems):
        x_s, y_s, _, _, fn = dataset.sample_one(support_size=10, query_size=0)
        problems.append([
            torch.Tensor(x_s),
            torch.Tensor(y_s)
        ])
        fns.append(fn)
    
    return problems, fns

train_problems, train_fns = make_dataset(train_dataset, num_problems)
valid_problems, valid_fns = make_dataset(valid_dataset, 10 * num_problems)

# --
# Setup model

encoder = Encoder()
alphas  = nn.Parameter(1 * torch.ones(num_problems))
betas   = nn.Parameter(1 * torch.ones(num_problems))

params = list(encoder.parameters()) + [alphas, betas]
opt    = torch.optim.LBFGS(params, max_iter=20)

# --
# Train

def _target_fn():
    opt.zero_grad()
    
    total_nll = 0
    for idx, (X, y) in enumerate(train_problems):
        alpha, beta = alphas[idx], betas[idx]
        phi = encoder(X)
        nll = compute_blr_nll(phi, y, alpha, beta)
        total_nll += nll
    
    total_nll = total_nll / len(train_problems)
    total_nll.backward()
    
    print(float(total_nll))
    
    return float(total_nll)

_ = opt.step(_target_fn)

# --
# Test (on training data)

mean_score = 0
for idx in range(len(alphas)):
    alpha, beta = alphas[idx], betas[idx]
    X, y = train_problems[idx]
    
    phi = encoder(X)
    
    score = BLR(alpha=alpha, beta=beta).score(phi, y)
    print(score)
    mean_score += score

mean_score /= num_problems
print('mean_score=%f' % mean_score)

# --
# Test (on validation data)

i = 2

X, y = valid_problems[i]
fn   = valid_fns[i]

valid_alpha = torch.Tensor([10]).requires_grad_()
valid_beta  = torch.Tensor([10]).requires_grad_()

opt2 = torch.optim.LBFGS([valid_alpha, valid_beta], lr=1, max_iter=20)
def _target_fn2():
    opt.zero_grad()
    phi = encoder(X)
    nll = compute_blr_nll(phi, y, valid_alpha, valid_beta)
    nll.backward()
    return float(nll)

opt2.step(_target_fn2)

phi   = encoder(X)
blr = BLR(alpha=valid_alpha, beta=valid_beta).fit(phi, y)

X_grid   = torch.linspace(*train_dataset.x_range, 100).view(-1, 1)
y_grid   = fn(X_grid)

phi_grid = encoder(X_grid)
mu_grid, sig_grid = blr.predict(phi_grid)

_ = plt.scatter(to_numpy(X).squeeze(), to_numpy(y).squeeze(), c='black')
_ = plt.plot(to_numpy(X_grid).squeeze(), to_numpy(y_grid).squeeze(), c='black', alpha=0.5)
_ = plt.plot(to_numpy(X_grid).squeeze(), to_numpy(mu_grid).squeeze(), c='red')
show_plot()


