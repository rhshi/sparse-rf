import numpy as np
from itertools import combinations
from sparse_rf.util import *


def make_X(d, m, dist=sphere):
    return dist((m, d))

def make_W(d, q, n=1, dist=normal, N=None):
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

def make_K(X, kernel=rbf, X_test=None):
    if X_test is None:
        K = np.zeros((len(X), len(X)))
        for i in range(len(X)):
            for j in range(i+1):
                val = kernel(X[i], X[j])
                K[i, j] = val
                K[j, i] = val
    else:
        assert X_test is not None
        m = X_test.shape[0]
        n = X.shape[0]
        K = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                K[i, j] = kernel(X_test[i], X[j])

    return K

