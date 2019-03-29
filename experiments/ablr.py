from rsub import *
from matplotlib import pyplot as plt

import numpy as np

import torch
from torch import nn

from hoof import dataset
from hoof.helpers import set_seeds, to_numpy

torch.set_default_tensor_type('torch.DoubleTensor')
torch.set_num_threads(1)

class ABLR(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, hidden_dim=32):
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
    sigma_t = (1 / alpha) * (phi @ phi.t()) + (1 / beta) * torch.eye(phi.shape[0])
    return 0.5 * (sigma_t.logdet() + y.t() @ torch.inverse(sigma_t) @ y).squeeze()


def blr_fit(phi, y, alpha, beta):
    d       = phi.shape[1]
    sig_inv = alpha * torch.eye(d) + beta * (phi.t() @ phi)
    sig     = torch.inverse(sig_inv)
    m       = beta * sig @ phi.t() @ y
    return m, sig


def blr_predict(phi, blr_model, beta):
    m, sig = blr_model
    mu_pred  = phi @ m
    sig_pred = 1 / beta + ((phi @ sig) * phi).sum(dim=-1)
    return mu_pred, sig_pred



dataset_name  = 'SinusoidDataset'
dataset_cls   = getattr(dataset, dataset_name)
train_dataset = dataset_cls()
valid_dataset = dataset_cls()

num_problems = 100

encoder = ABLR()
alphas  = nn.Parameter(10 * torch.ones(num_problems))
betas   = nn.Parameter(10 * torch.ones(num_problems))

train_problems, train_fns = [], []
for _ in range(num_problems):
    x_s, y_s, _, _, fn = train_dataset.sample_one(support_size=10, query_size=0)
    train_problems.append([
        torch.Tensor(x_s),
        torch.Tensor(y_s)
    ])
    train_fns.append(fn)


valid_problems, valid_fns = [], []
for _ in range(num_problems):
    x_s, y_s, _, _, fn = valid_dataset.sample_one(support_size=10, query_size=0)
    valid_problems.append([
        torch.Tensor(x_s),
        torch.Tensor(y_s)
    ])
    valid_fns.append(fn)

# >>

from tqdm import trange

encoder = ABLR()
alphas  = nn.Parameter(torch.ones(num_problems))
betas   = nn.Parameter(torch.ones(num_problems))

num_samples = 10000

params = list(encoder.parameters()) + [alphas, betas]
opt    = torch.optim.Adam(params)
opt.zero_grad()

hist = []
gen = trange(num_samples)
for batch_idx in gen:
    x_support, y_support, x_query, y_query, _ = train_dataset.sample_batch(
        batch_size=1,
        support_size=10,
        query_size=0,
    )
    
    x_support = torch.Tensor(x_support)
    y_support = torch.Tensor(y_support)
    
    phi = encoder(x_support)
    
    
    loss = compute_blr_nll(phi[0], y_support[0], alphas[i], betas[i])
    loss.backward()
    
    if not batch_idx % 128:
        opt.step()
        opt.zero_grad()
    
    blr_model = blr_fit(phi[0], y_support[0], alphas[i], betas[i])
    mu, _     = blr_predict(phi[0], blr_model, betas[i])
    
    hist.append(float((y_support[0] - mu).pow(2).mean()))
    
    if not batch_idx % 128:
        gen.set_postfix(loss='%0.8f' % np.mean(hist[-128:]))




# <<








params = list(encoder.parameters()) + [alphas, betas]
opt    = torch.optim.LBFGS(params, lr=0.1, max_iter=40)

def _target():
    opt.zero_grad()
    
    res = [compute_blr_nll(encoder(X), y, alphas[i], betas[i]) for i, (X, y) in enumerate(train_problems)]
    nll = torch.mean(torch.stack(res))
    nll.backward()
    
    print(float(nll))
    return float(nll)

opt.step(_target)

all_loss = []
for i, (X, y) in enumerate(valid_problems):
    alpha, beta = alphas[i], betas[i]
    
    phi       = encoder(X)
    blr_model = blr_fit(phi, y, alpha, beta)
    mu_grid, sig_grid = blr_predict(phi, blr_model, beta)
    
    loss = float((mu_grid - y).pow(2).mean())
    all_loss.append(loss)

print(np.mean(all_loss))


i     = np.random.choice(len(valid_problems))
X, y  = valid_problems[i]
fn    = valid_fns[i]
alpha = alphas[i]
beta  = betas[i]

phi       = encoder(X)
blr_model = blr_fit(phi, y, alpha, beta)

X_grid   = np.linspace(*train_dataset.x_range, 100).reshape(-1, 1)
y_grid   = fn(X_grid)

phi_grid = encoder(torch.Tensor(X_grid))
mu_grid, sig_grid = blr_predict(phi_grid, blr_model, beta)

mu_grid  = to_numpy(mu_grid).squeeze()
sig_grid = to_numpy(sig_grid).squeeze()

_ = plt.scatter(X.squeeze(), y.squeeze())
_ = plt.plot(X_grid.squeeze(), mu_grid, c='red')
_ = plt.plot(X_grid.squeeze(), y_grid.squeeze(), c='black')
_ = plt.fill_between(X_grid.squeeze(), mu_grid - 1.96 * np.sqrt(sig_grid), mu_grid + 1.96 * np.sqrt(sig_grid), alpha=0.2, color='red')
show_plot()



