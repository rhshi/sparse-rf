import numpy as np
from sparse_rf.modules import *
from sparse_rf.util import *
import random
from sparse_rf.algs.core import *
import gc

def prune(w, A_train, A_test, y_train, y_test, method, per=20):
    thre = np.percentile(np.abs(w), per)
    idx = np.abs(w) > thre
    # output mse after pruninig--bad results
    y_pred = A_test[:, idx] @ w[idx]
    mse_prune = 1/len(y_test) * np.linalg.norm(y_pred - y_test)**2
    # print(f"mse after prunining with threshold {thre}: {mse_prune}")

    # retraining--IMP--good resultS
    A_trains = A_train[:, idx]
    A_tests = A_test[:, idx]
    w_prune = method(A_trains, y_train)
    y_preds = A_tests @ w_prune
    mse_retrain = 1/len(y_test) * np.linalg.norm(y_preds - y_test)**2
    print(f"mse after training with {w_prune.shape}")

    return A_trains, A_tests, w_prune, mse_prune, mse_retrain

def prune_total(A_train_sparse, A_test_sparse, y_train, y_test, c_l2_sparse, method, step=30, per=20):
    A_trains, A_tests, w_prune = A_train_sparse, A_test_sparse, c_l2_sparse
    mse_prune = []
    mse_retrain = []
    w_length = []

    for i in range(step):
        A_trains, A_tests, w_prune, msep, mser = prune(w_prune, A_trains, A_tests, y_train, y_test, method, per)
        mse_prune.append(msep)
        mse_retrain.append(mser)
        w_length.append(w_prune.shape[0])

    return w_length, mse_prune, mse_retrain

# double descent curve plot with randomly sampled weights
def reinit(X_train, X_test, y_train, y_test, d, q, method, w_length, active=fourier, dist=normal):
    res_reinit = []
     # new initialization
    for wl in w_length:
        W_sparse = make_W(d, q, N=wl//2, dist=dist)
        A_train_sparse = make_A(X_train, W_sparse, active=active)
        A_test_sparse = make_A(X_test, W_sparse, active=active)
        # print(A_train_sparse.shape, wl)

        c = method(A_train_sparse, y_train)
        res = np.linalg.norm(A_test_sparse@c-y_test)**2 / len(y_test)
        res_reinit.append(res)
    return res_reinit 



def test_perfomance(d, qs, N, m, func, ratio_train, seeds, dist=normal, active=fourier):
    X = make_X(d, m, dist=uniform)
    X_train = X[:int(m*ratio_train), :]
    X_test = X[int(m*ratio_train):, :]

    y = np.array(list(map(func, X)))
    y_train = y[:int(m*ratio_train)]
    y_test = y[int(m*ratio_train):]

    results_l1 = []
    results_l2 = []
    prune_result = []
    w_result = []
    for q in qs:
        res_l1 = []
        res_l2 = []
        res_prune = []
        w_len = []
        
        for seed in seeds:
            random.seed(seed)
            np.random.seed(seed)

            W = make_W(d, q, N=N, dist=normal)
            A_train = make_A(X_train, W, active=active)
            A_test = make_A(X_test, W, active=active)

            c_l1 = min_l1(A_train, y_train)
            c_l2 = min_l2(A_train, y_train)

            res_l1.append(np.linalg.norm(A_test@c_l1 - y_test) / np.linalg.norm(y_test))
            res_l2.append(np.linalg.norm(A_test@c_l2 - y_test) / np.linalg.norm(y_test))

            w_length, mse_prune, mse_retrain = prune_total(A_train, A_test, y_train, y_test, c_l2, method=min_l2, step=30, per=20)
            idx = np.argmin(mse_retrain)
            res_prune.append(mse_retrain[idx])
            w_len.append(w_length[idx])


            del W
            del A_train
            del A_test
            del c_l1
            del c_l2

            gc.collect()

        results_l1.append(res_l1)
        results_l2.append(res_l2)
        prune_result.append(res_prune)
        w_result.append(w_len)


    return X_train, X_test, y_train, y_test, results_l1, results_l2, prune_result, w_result, w_length
