import numpy as np
from itertools import combinations
from sparse_rf.util import *


def make_X(d, m, dist=sphere):
    return dist((m, d))

def make_W(d, q, n, dist=normal, N=None):
    if N is not None:
        W = np.zeros((N, d))
        i = 0
        while (i<N):
            w = dist(d)
            ind = np.random.choice(d, d-q, replace=False)
            w[ind] = 0
            W[i, :] = w
            i += 1
    else:
        num_supports = comb(d, q)
        W = np.zeros((n*num_supports, d))
        inds = combinations(range(d), d-q)
        for i in range(num_supports):
            ind = next(inds)
            for j in range(n):
                w = dist(d)
                w[list(ind)] = 0
                W[i*n+j, :] = w
    return W

def make_A(X, W, active=relu):
    return active(np.matmul(X, W.T))

    

