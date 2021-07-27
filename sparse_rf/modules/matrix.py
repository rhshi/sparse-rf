import numpy as np
from itertools import combinations
from functools import partial
from sparse_rf.util import *


def make_X(d, m, dist=uniform):
    return dist((m, d))

def make_W(d, q, N, dist=normal, scale=1):
    num_supports = comb(d, q)
    if num_supports > N:
        W = np.zeros((N, d))
        inds_track = None
        i = 0
        while (i<N):
            w = partial(dist, stdev=scale)(d)
            ind = np.random.choice(d, d-q, replace=False)
            w[ind] = 0
            W[i, :] = w
            i += 1
    else:
        n = N // num_supports
        Nreal = n * num_supports
        W = np.zeros((Nreal, d))
        inds_track = np.zeros((num_supports, q))
        inds = combinations(range(d), d-q)
        for i in range(num_supports):
            ind = next(inds)
            inds_track[i] = np.setdiff1d(range(d), ind, assume_unique=True)
            for j in range(n):
                w = partial(dist, stdev=scale)(d)
                w[list(ind)] = 0
                W[i*n+j, :] = w
    return W, inds_track

def make_A(X, W, active=fourier):
    temp = active(np.matmul(X, W.T))
    if active == fourier:
        return temp / np.sqrt(temp.shape[-1]/2)
    else:
        return temp

def make_K(X, kernel=rbf, X_test=None):
    if X_test is None:
        return kernel(X, X)
    else:
        assert X_test is not None
        return kernel(X_test, X)

