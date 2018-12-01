#!/usr/bin/env python

"""
    svm_experiment.py
"""

from rsub import *
from matplotlib import pyplot as plt

import numpy as np
from scipy import sparse

from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

np.random.seed(123)


n_toks = 1000
n_obs  = 100

p = 10 ** np.random.uniform(-5, 0, n_toks).reshape(1, -1)
q = 10 ** np.random.uniform(-5, 0, n_toks).reshape(1, -1)

pos = np.random.uniform(0, 1, (n_obs, n_toks)) < p
neg = np.random.uniform(0, 1, (n_obs, n_toks)) < q

X = np.vstack([pos, neg])
X /= np.sqrt((X ** 2).sum(axis=-1, keepdims=True))
y   = np.hstack([
    np.ones(pos.shape[0]),
    np.zeros(neg.shape[0])
])



# --

from sklearn.linear_model import SGDClassifier

def run_experiment(dim=5, nobs=1000, pos_mean=0, pos_std=1, neg_mean=1, neg_std=1):
    pos = np.random.normal(pos_mean, pos_std, (nobs, dim))
    neg = np.random.normal(neg_mean, neg_std, (nobs, dim))
    X   = np.vstack([pos, neg]).astype(float)
    X /= np.sqrt((X ** 2).sum(axis=-1, keepdims=True))
    y   = np.hstack([
        np.ones(pos.shape[0]),
        np.zeros(neg.shape[0])
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5)
    
    all_accs = []
    alphas = np.arange(-5, 1, 0.1)
    for alpha in alphas:
        preds = SGDClassifier(alpha=10 ** alpha).fit(X_train, y_train).predict(X_test)
        acc = (y_test == preds).mean()
        all_accs.append(acc)
        
    return alphas, all_accs


for _ in range(5):
    alphas, all_accs = run_experiment()
    _ = plt.plot(alphas, all_accs, c='red', alpha=0.3)

for _ in range(5):
    Cs, all_accs = run_experiment(neg_std=2)
    _ = plt.plot(Cs, all_accs, c='green', alpha=0.3)

show_plot()




