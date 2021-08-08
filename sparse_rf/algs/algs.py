import numpy as np
from sparse_rf.modules import *
from sparse_rf.util import *
import random
from sparse_rf.algs.core import *
import gc
from namedlist import namedlist

BestModel = namedlist("BestModel", ["n_best", "id_list", "min_val", "W", "q", "w", "A_train", "inds_track", "thresh"])

def generate_m(X, q, N):
    d = X.shape[-1]
    scale = 1/np.sqrt(q)
    W, inds_track = make_W(d, q, N, scale=scale)
    return make_A(X, W), W, inds_track

def shrimp(X, Y, numPartsKFoldCV=10, orderCands=list(range(1, 6)), N=10000, step=100, per=0.25, l=0, random=False):
    m = X.shape[0]
    shuffleOrder = np.random.permutation(m)
    X = X[shuffleOrder, :]
    Y = Y[shuffleOrder]

    numOrderCands = len(orderCands)
    bestmodel = BestModel(-1, -1, np.inf, -1, -1, -1, 0, -1, -1)
    for i in range(numOrderCands):
        q = orderCands[i]
        A_train, W, inds_track = generate_m(X, q, N)
        n_best, id_list, min_mse, w, thresh = validate(A_train, Y, numPartsKFoldCV, step, per, l, random)
        if min_mse < bestmodel.min_val:
            bestmodel.inds_track = inds_track
            bestmodel.n_best = n_best
            bestmodel.id_list = id_list
            bestmodel.min_val = min_mse
            bestmodel.W = W
            bestmodel.q = q
            bestmodel.w = w
            bestmodel.A_train = A_train
            bestmodel.thresh = thresh

    A = bestmodel.A_train
    A_train_final = A[:, bestmodel.id_list]
    w = l2(A_train_final, Y)
    bestmodel.A_train = 0
    bestmodel.w = w 

    return bestmodel

def validate(A, Y, numPartsKFoldCV, step, per, l, random):
    m = A.shape[0]
    cvIter = 1

    testStartIdx = (cvIter-1)*m//numPartsKFoldCV
    testEndIdx = cvIter*m//numPartsKFoldCV
    trainIdxs = list(range(testStartIdx)) + list(range(testEndIdx, m))
    testIdxs = list(range(testStartIdx, testEndIdx))
    Atr = A[trainIdxs, :]
    Aval = A[testIdxs, :]
    Ytr = Y[trainIdxs]
    Yval = Y[testIdxs]
    
    w_len, mse_rec, list_rec, ww, thresh = shrimp_prune(Atr, Aval, Ytr, Yval, step, per, l, random)
    min_mse_id = np.argmin(mse_rec)
    min_mse = mse_rec[min_mse_id]
    n_best = w_len[min_mse_id]
    id_list = list_rec[str(n_best)]
    w = ww[str(n_best)]

    return n_best, id_list, min_mse, w, thresh

def shrimp_prune(A_train, A_test, y_train, y_test, step, per, l, random):
    w_prune = l2(A_train, y_train, l=0)
    y_preds = A_test@w_prune
    mse = np.sum((y_test-y_preds)**2) / len(y_test)

    w_len = [len(w_prune)]
    mse_record = [mse]

    ind_list = np.array(range(len(w_prune)))
    list_rec = {str(len(w_prune)): ind_list}
    ww = {str(len(w_prune)): w_prune}
    # thresh = {str(len(w_prune)): 0}
    thresh = {}

    for i in range(step):

        old_len = len(w_prune)

        if old_len == 1:
            break
    
        A_train, A_test, w_prune, mse, ind_list, thre = prune_os(w_prune, A_train, A_test, y_train, y_test, ind_list, per, random)
        w_len.append(len(w_prune))
        mse_record.append(mse)
        list_rec[str(len(w_prune))] = ind_list
        ww[str(len(w_prune))] = w_prune
        thresh["{} -> {}".format(old_len, len(w_prune))] = thre

    return w_len, mse_record, list_rec, ww, thresh

def prune_os(w, A_train, A_test, y_train, y_test, ind_list, per, random):
    if not random:
        thre = np.quantile(np.abs(w), per)
        idx = abs(w) > thre
        high = np.min(np.abs(w[idx]))
        low = np.min(np.abs(w[np.abs(w) <= thre]))
    else:
        high = None
        low = None
        idx = np.full(len(w), True)
        idx[:int(per*len(idx))] = False
        np.random.shuffle(idx)

    new_list = ind_list[idx]
    a = np.array(range(len(w)))[idx]

    A_trains = A_train[:, a]
    A_tests = A_test[:, a]
    w_prune = l2(A_trains, y_train)
    y_preds = A_tests@w_prune
    mse = np.sum((y_test-y_preds)**2) / len(y_test)

    return A_trains, A_tests, w_prune, mse, new_list, (low, high)
