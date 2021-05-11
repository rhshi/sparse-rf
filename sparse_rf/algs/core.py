import numpy as np
from spgl1 import spg_bpdn


def min_l1(A, y, eta=0):
    return spg_bpdn(A, y, sigma=eta)[0]

def min_l2(A, y):
    return np.linalg.pinv(A)@y

def min_wl2(A, y, D):
    return D@np.linalg.pinv(A@D)@y

