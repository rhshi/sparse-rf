import numpy as np
from dists import *
from activations import *


def make_X(d, m, dist=sphere):
    return dist((m, d))

def make_W(d, N, q, dist=normal):
    W = np.zeros((N, d))
    i = 0
    while (i<N):
        w = dist(d)
        ind = np.random.choice(d, d-q, replace=False)
        w[ind] = 0
        W[i, :] = w
        i += 1

    return W

def make_A(X, W, active=relu):
    return active(np.matmul(X, W.T))

    

