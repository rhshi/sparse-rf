import numpy as np
from itertools import combinations
from sparse_rf.util.util import comb

def relu(x):
    return np.maximum(0, x)

def fourier(x):
    return np.concatenate((np.cos(x), np.sin(x)), axis=-1)

def identity(x):
    return x

def rbf(x, y, scale=1):
    return np.exp(-scale**2 * np.sum((x-y)**2) / 2)

def sparse_rbf(x, y, d, q, scale=1):
    inds = combinations(range(d), q)
    val = 0
    for ind in inds:
        val += rbf(x[list(ind)], y[list(ind)], scale=scale)
    
    return val / comb(d, q)