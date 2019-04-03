#!/usr/bin/env python

"""
    exline/bulk2/rf_bulk.py
"""

import sys
def warn(*args, **kwargs): pass

import warnings
warnings.warn = warn

import numpy as np
np.warnings.filterwarnings('ignore')

import os
import sys
import json
import argparse
import openml
from time import time
from tqdm import tqdm
from uuid import uuid4
from datetime import datetime
from joblib import Parallel, delayed

from sklearn.base import clone
from sklearn.model_selection import ParameterGrid

from exline.helpers import with_timeout
from exline.preprocessing.featurization import DataFrameMapper
from exline.modeling.metrics import metrics, classification_metrics
from exline.modeling.forest import ForestConfigSpace, EitherForestClassifier, EitherForestRegressor

from exline.utils import get_git_hash

sys.path.append('/home/bjohnson/projects/exline/experiments/bulk2/')
from dataset_util import load_openml_problem


def _raw_eval_grid_point(random_state, config, prob_name, dataset, target_budget, n_jobs):
    Xf_train, Xf_test, y_train, y_test, target_metric = dataset
    
    ForestEstimator = EitherForestClassifier
    
    model = ForestEstimator(
        oob_score=True,
        bootstrap=True,
        n_jobs=n_jobs,
        n_estimators=target_budget,
        random_state=random_state,
        **config
    )
    
    config['n_estimators'] = target_budget
    
    model = model.fit(Xf_train, y_train)
        
    if target_metric in classification_metrics:
        pred_valid = model.classes_[model.oob_decision_function_.argmax(axis=-1)]
    
    pred_test = model.predict(Xf_test)
    
    config_id = str(uuid4())
    
    return {
        "config_id"     : config_id,
        "target_metric" : target_metric,
        "valid_score"   : metrics[target_metric](y_train, pred_valid),
        "test_score"    : metrics[target_metric](y_test, pred_test),
    }


def eval_grid_point(random_state, config, target_budget, prob_name, dataset, git_hash, seed, n_jobs=1):
    t = time()
    try:
        res = _raw_eval_grid_point(
            random_state=random_state,
            config=config,
            prob_name=prob_name,
            dataset=dataset,
            target_budget=target_budget,
            n_jobs=n_jobs,
        )
    except Exception as e:
        print('error %s' % prob_name, file=sys.stderr)
        print(e, file=sys.stderr)
        res = {
            "config_id"     : None,
            "valid_score"   : None,
            "test_score"    : None,
            "target_metric" : None,
        }
    
    res.update({
        "prob_name" : prob_name,
        "config"    : config,
        "elapsed"   : time() - t,
        "_run_info" : {
            "random_state"  : random_state,
            "git_hash"      : git_hash,
            "timestamp"     : datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
            "seed"          : seed,
        }
    })
    
    # print(json.dumps(res))
    # sys.stdout.flush()
    return res


prob_names = valid_dataset.task_ids
datasets   = {}
for prob_name in tqdm(prob_names):
    datasets[prob_name] = load_openml_problem(prob_name, mode='exline')

cs = ForestConfigSpace()

def cs2vec(cfg):
    return np.array([
        cfg['max_features'],
        np.log10(cfg['min_samples_leaf']),
        np.log10(cfg['min_samples_split']),
        cfg['class_weight'] is None,
        cfg['class_weight'] == "balanced",
        cfg['class_weight'] == "balanced_subsample",
        cfg['estimator'] == "ExtraTrees",
        cfg['estimator'] == "RandomForest",
    ])

def vec2cs(vec):
    tmp = {
        'class_weight'      : vec[0],
        'estimator'         : vec[1],
        'max_features'      : vec[2],
        'min_samples_leaf'  : int(10 ** vec[3]),
        'min_samples_split' : int(10 ** vec[4]),
    }
    if tmp['class_weight'] == 'none':
        tmp['class_weight'] = None
    
    return tmp


def cs2gp(cfg):
    cfg['min_samples_leaf']  = np.log10(cfg['min_samples_leaf'])
    cfg['min_samples_split'] = np.log10(cfg['min_samples_split'])
    return [cfg[p[6:]] for p in gp_param_cols]


def gp_live(prob_name, sub, dimensions, max_steps=40):
    opt = Optimizer(
        dimensions=dimensions,
        base_estimator="GP",
        n_initial_points=5,
        acq_func="EI",
        acq_optimizer="sampling"
    )
    opt._X_cand = None
    
    traj       = []
    all_output = []
    
    t = time()
    for iteration in range(max_steps):
        
        next_x  = opt.ask()
        next_cs = vec2cs(next_x)
        
        output = eval_grid_point(
            random_state  = int(np.random.randint(2 ** 16)),
            config        = next_cs,
            target_budget = 128,
            prob_name     = prob_name,
            dataset       = datasets[prob_name],
            git_hash      = '***',
            seed          = int(np.random.randint(2 ** 16)),
            n_jobs        = 40,
        )
        
        next_y = -1 * output['valid_score']
        _ = opt.tell(next_x, next_y)
        
        traj.append(next_y)
        all_output.append(output)
    
    return np.array(traj), all_output


from copy import copy
def alpaca_bo_live(model, prob_name, num_rounds=20, burnin_size=2, explore_eps=0.001, acq='ei', adjust_alpha=False):
    model = deepcopy(model)
    
    all_output = []
    
    # --
    # Burnin
    
    # print('burning in w/ %d function evals' % burnin_size, file=sys.stderr)
    cs_burn = cs.sample_configurations(n=burnin_size)
    x_burn  = np.stack([cs2vec(xx) for xx in cs_burn])
    
    output_burn = [eval_grid_point(
        random_state  = int(np.random.randint(2 ** 16)),
        config        = config,
        target_budget = 128,
        prob_name     = prob_name,
        dataset       = datasets[prob_name],
        git_hash      = '***',
        seed          = int(np.random.randint(2 ** 16)),
        n_jobs        = 40,
    ) for config in cs_burn]
    
    all_output += output_burn
    y_burn = np.array([- xx['valid_score'] for xx in output_burn]).reshape(-1, 1)
    
    x_visited, y_visited = x_burn, y_burn
    
    # --
    # Alternative: Hotstart
    
    # sel = np.random.choice(sub.shape[0], 100)
    # x_visited = np.vstack(sub.Xf.values)[sel]
    # y_visited = -1 * sub.valid_score.values[sel].reshape(-1, 1)
    
    # --
    # BO iterations
    # print('running %d BO iterations' % (num_rounds - burnin_size), file=sys.stderr)
    
    traj = np.sort(y_visited.squeeze())[::-1]
    for iteration in range(num_rounds - burnin_size):
        
        if adjust_alpha and (iteration in [8, 16, 32, 64, 128, 256, 512]):
            model.blr.alpha /= 2
        
        cs_cand = cs.sample_configurations(n=5000)
        x_cand  = np.stack([cs2vec(xx) for xx in cs_cand])
        
        inp = list2tensors((x_visited, y_visited, x_cand), cuda=model.is_cuda)
        mu, sig, _ = model(*inp)
        mu, sig = tensors2list((mu, sig), squeeze=True)
        
        # Expected improvement
        ei       = gaussian_ei(mu, sig, incumbent=y_visited.min())
        best_idx = ei.argmax()
        
        next_x  = x_cand[best_idx]
        next_cs = cs_cand[best_idx]
        
        output = eval_grid_point(
            random_state  = int(np.random.randint(2 ** 16)),
            config        = copy(next_cs),
            target_budget = 128,
            prob_name     = prob_name,
            dataset       = datasets[prob_name],
            git_hash      = '***',
            seed          = int(np.random.randint(2 ** 16)),
            n_jobs        = 60,
        )
        all_output.append(output)
        
        
        next_y = - output['valid_score']
        
        x_visited = np.vstack([x_visited, next_x])
        y_visited = np.vstack([y_visited, next_y])
        
        traj = np.hstack([traj, [y_visited.min()]])
    
    return traj, all_output


model = model.cuda()

all_results = {}

for it in range(10):
    for prob_name in task_ids:
        sub = df[df.task_id == prob_name]
        
        model_traj, model_output = alpaca_bo_live(model, prob_name, num_rounds=1, adjust_alpha=True)
        gp_traj, gp_output       = gp_live(prob_name, sub, max_steps=1, dimensions=dimensions)
        
        print(it, prob_name, model_traj.min(), gp_traj.min(), -sub.valid_score.max())
        
        all_results[(it, prob_name)] = {
            "model_traj"   : model_traj,
            "model_output" : model_output,
            "gp_traj"      : gp_traj,
            "gp_output"    : gp_output,
        }

