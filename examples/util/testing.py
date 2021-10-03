import numpy as np  
from examples.util.metrics import shrimp_test, make_l2_loss
from sparse_rf.util import *
from sparse_rf.modules import make_A, make_W
from sparse_rf.algs import shrimp, l2, l1, sindy
from collections import namedtuple

shrimp_v_random_ = namedtuple("shrimp_v_random_", 
    [
        'lengths', 
        'err_l2',
        'err_shrimp',
        'err_r',
        'thresh_l',
        'thresh_u',
        'train_mse_s',
        'val_mse_s',
        'train_mse_r',
        'val_mse_r',
        'eigs_s_min',
        'eigs_s_max',
        'eigs_r_min',
        'eigs_r_max',
        'n_best_s',
        'n_best_r'
    ]
)


def shrimp_v_random(trials, Xtr, Ytr, Xte, Yte, q, N, l, scale=None, verbose=0):
    errs_l2 = []
    errs_shrimp = []
    errs_r = []

    threshs = {'l': [], 'u': []}
    n_bests_s = []
    n_bests_r = []

    all_train_mses_s = []
    all_train_mses_r = []
    all_val_mses_s = []
    all_val_mses_r = []

    all_eigs_s = {"min": [], "max": []}
    all_eigs_r = {"min": [], "max": []}

    for _ in range(trials):
        best_model, train_mses, val_mses, lengths = shrimp(Xtr, Ytr, orderCands=[q], verbose=verbose, N=N, l=l, scale=scale)
        w_shrimp, Ate_s = shrimp_test(best_model, Xte)
        Atr = make_A(Xtr, best_model.W)
        w_l2 = l2(Atr, Ytr, l=l)
        best_model_r, train_mses_r, val_mses_r, _ = shrimp(Xtr, Ytr, orderCands=[q], verbose=verbose, N=N, random=True, l=l, scale=scale)
        w_r, Ate_r = shrimp_test(best_model_r, Xte)

        l2_loss_s = make_l2_loss(Ate_s, Yte)
        l2_loss_r = make_l2_loss(Ate_r, Yte)

        errs_shrimp.append(l2_loss_s(w_shrimp))
        errs_l2.append(l2_loss_s(w_l2))
        errs_r.append(l2_loss_r(w_r))

        thresh = list(zip(*list(best_model.thresh.values())))
        threshs['l'].append(thresh[0])
        threshs['u'].append(thresh[1])

        all_train_mses_s.append(train_mses[q])
        all_train_mses_r.append(train_mses_r[q])
        all_val_mses_s.append(val_mses[q])
        all_val_mses_r.append(val_mses_r[q])

        n_bests_s.append(best_model.n_best)
        n_bests_r.append(best_model_r.n_best)
        all_eigs_s["min"].append(best_model.eigs["min"])
        all_eigs_s["max"].append(best_model.eigs["max"])
        all_eigs_r["min"].append(best_model_r.eigs["min"])
        all_eigs_r["max"].append(best_model_r.eigs["max"])

    thresh_l = np.mean(threshs['l'], axis=0)
    thresh_u = np.mean(threshs['u'], axis=0)
    train_mse_s = np.mean(all_train_mses_s, axis=0)
    train_mse_r = np.mean(all_train_mses_r, axis=0)
    val_mse_s = np.mean(all_val_mses_s, axis=0)
    val_mse_r = np.mean(all_val_mses_r, axis=0)

    eigs_s_min = np.real(np.mean(all_eigs_s["min"], axis=0))
    eigs_s_max = np.real(np.mean(all_eigs_s["max"], axis=0))
    eigs_r_min = np.real(np.mean(all_eigs_r["min"], axis=0))
    eigs_r_max = np.real(np.mean(all_eigs_r["max"], axis=0))

    err_l2 = np.mean(errs_l2)
    err_shrimp = np.mean(errs_shrimp)
    err_r = np.mean(errs_r)

    n_best_s = np.mean(n_bests_s)
    n_best_r = np.mean(n_bests_r)

    return shrimp_v_random_(lengths, err_l2, err_shrimp, err_r, thresh_l, thresh_u, train_mse_s, train_mse_r, val_mse_s, val_mse_r, eigs_s_min, eigs_s_max, eigs_r_min, eigs_r_max, n_best_s, n_best_r)

growth_ = namedtuple("growth_", 
    [
        "err_g",
        "g_train_mse",
        "g_val_mse",
        "n_best",
        "lengths",
        "g_eigs_min",
        "g_eigs_max"
    ]
)

def growth(trials, Xtr, Ytr, Xte, Yte, N_, q, l, inter=1):
    lengths = range(0, N_, inter)

    m, d = Xtr.shape

    all_g_train_mses = []
    all_g_val_mses = []
    errs_g = []
    best_lens = []
    all_eigs_min = []
    all_eigs_max = []

    for _ in range(trials):
        g_train_mses = []
        g_val_mses = []
        eigs_min = []
        eigs_max = []
        best_W = -1
        best_val = np.inf
        best_c = -1
        Xtr1 = Xtr[:int(m*0.9), :]
        Xval = Xtr[int(m*0.9):, :]
        Ytr1 = Ytr[:int(m*0.9)]
        Yval = Ytr[int(m*0.9):]
        for g in lengths:
            W, _ = make_W(d, q, N=g, scale=1/np.sqrt(q), sample=False)
            Atr1 = make_A(Xtr1, W)
            Aval = make_A(Xval, W)
            cg = l2(Atr1, Ytr1, l=l)
            val = np.sum(Yval-Aval@cg)**2/len(Yval)
            g_train_mses.append(np.sum(Ytr1-Atr1@cg)**2/len(Ytr1))
            g_val_mses.append(val)

            if Atr1.shape[0] < Atr1.shape[1]:
                eigs_ = np.sort(np.real(np.linalg.eigvals(Atr1@Atr1.T)))
            else:
                eigs_ = np.sort(np.real(np.linalg.eigvals(Atr1.T@Atr1)))
            eigs_min.append(eigs_[0])
            eigs_max.append(eigs_[1])


            if val < best_val:
                best_val = val
                best_W = W
                best_c = cg

        Ate = make_A(Xte, best_W)
        err_g = np.sum(Yte-Ate@best_c)**2/len(Yte)

        all_g_train_mses.append(g_train_mses)
        all_g_val_mses.append(g_val_mses)
        errs_g.append(err_g)
        best_lens.append(len(best_c))
        all_eigs_min.append(eigs_min)
        all_eigs_max.append(eigs_max)

    g_train_mse = np.mean(all_g_train_mses, axis=0)
    g_val_mse = np.mean(all_g_train_mses, axis=0)
    g_eigs_min = np.real(np.mean(all_eigs_min, axis=0))
    g_eigs_max = np.real(np.mean(all_eigs_max, axis=0))
    err_g_ = np.mean(errs_g)
    n_best = np.mean(best_lens)

    return growth_(err_g_, g_train_mse, g_val_mse, n_best, lengths, g_eigs_min, g_eigs_max)