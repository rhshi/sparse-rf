import numpy as np

def w_norm(w, l=1):
    return np.sum(np.abs(w)**l)