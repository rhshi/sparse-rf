import numpy as np
from sparse_rf.modules import make_A

def w_norm(w, l=1):
    return np.sum(np.abs(w)**l)

def make_l2_loss(A, y):
    def l2_loss(w):
        return np.sum((y-A@w)**2)/len(y)
    return l2_loss

def shrimp_test(best_model, Xte):
    w = np.zeros(2*best_model.W.shape[0])
    w[best_model.id_list] = best_model.w
    Ate = make_A(Xte, best_model.W)
    return w, Ate