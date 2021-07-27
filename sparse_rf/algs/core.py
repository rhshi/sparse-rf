from namedlist import _nl_count
import numpy as np
from spgl1 import spg_bpdn


def l1(A, y, eta=0):
    return spg_bpdn(A, y, sigma=eta)[0]

def l2(A, y, l=0):
    if l == 0:
        return np.linalg.pinv(A)@y
    elif l > 0:
        m, N = A.shape
        if m > N:
            return np.linalg.inv(A.T@A/m+l*np.eye(N))@A.T@y/m
        else:
            return A.T@np.linalg.inv(A@A.T/m+l*np.eye(m))@y/m 

def wl2(A, y, D):
    return D@np.linalg.pinv(A@D)@y

