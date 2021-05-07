import numpy as np
from itertools import combinations
from sparse_rf.util.util import comb
from sklearn.metrics.pairwise import rbf_kernel

def relu(x):
    return np.maximum(0, x)

def fourier(x):
    return np.concatenate((np.cos(x), np.sin(x)), axis=-1)

def cosine(x):
    return np.cos(x)

def sine(x):
    return np.sin(x)

def identity(x):
    return x

def rbf(x, y, scale=1):
    return rbf_kernel(x, y, gamma=scale**2/2)

def sparse_rbf(x, y, d, q, scale=1):
    inds = combinations(range(d), q)
    val = 0
    for ind in inds:
        val += rbf(x[:, list(ind)], y[:, list(ind)], scale=scale)
    
    return val / comb(d, q)