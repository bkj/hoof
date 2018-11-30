#!/usr/bin/env python

"""
    dataset.py
"""

import numpy as np
from copy import copy

def _make_f(amp, phase, freq):
    def _f(x, noise=True):
        y = amp * np.sin(freq * x + phase)
        # if noise:
            # y += np.random.normal(0, self.noise_std, x.shape)
        
        return y
    
    return _f

class SinusoidDataset:
    def __init__(self, amp_range=[0.1, 5.0], phase_range=[0, 3.14], 
        freq_range=[0.999, 1.0], x_range=[-5, 5], sigma_eps=0.02):
        
        self.amp_range   = amp_range
        self.phase_range = phase_range
        self.freq_range  = freq_range
        self.x_range     = x_range
        self.noise_std   = np.sqrt(sigma_eps)
        
    def sample(self, n_funcs, train_samples, test_samples):
        
        x_c = np.random.uniform(*self.x_range, (n_funcs, train_samples, 1))
        x   = np.random.uniform(*self.x_range, (n_funcs, test_samples, 1))
        
        y_c = np.zeros((n_funcs, train_samples, 1))
        y   = np.zeros((n_funcs, test_samples, 1))
        
        funcs = []
        for i in range(n_funcs):
            
            amp   = np.random.uniform(*self.amp_range)
            phase = np.random.uniform(*self.phase_range)
            freq  = np.random.uniform(*self.freq_range)
            
            _f     = _make_f(amp, phase, freq)
            y_c[i] = _f(x_c[i])
            y[i]   = _f(x[i])
            
            funcs.append(copy(_f))
        
        return x_c, y_c, x, y, funcs

if __name__ == "__main__":
    np.random.seed(123)
    d = SinusoidDataset()
    d.sample(n_funcs=2, train_samples=4, test_samples=1)



