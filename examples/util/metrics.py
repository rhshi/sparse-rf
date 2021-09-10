import numpy as np

def w_norm(w, l=1):
    return np.sum(np.abs(w)**l)

def make_l2_loss(A, y):
    def l2_loss(w):
        return np.sum((y-A@w)**2)/len(y)
    return l2_loss