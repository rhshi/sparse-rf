import numpy as np
from sparse_rf.util import *
from sparse_rf.algs import make_X
from scipy.io import loadmat

def getDataset(dataset, d=10, m=1000, train_ratio=0.7, noise=0, dist=uniform):
    if isinstance(dataset, str):
        if dataset == "speech":
            L = np.loadtxt("./datasets/parkinson21.txt", delimiter=",")
            L = shuffleData(L)
            attrs = list(range(14)) + list(range(19, 26))
            label = 14
            trIdxs = list(range(520))
            teIdxs = list(range(521, 1040))
            Xtr, Ytr, Xte, Yte = partitionData(L, attrs, label, trIdxs, teIdxs)

        elif dataset == "propulsion":
            L = np.loadtxt("./datasets/propulsion.txt")
            L = shuffleData(L)
            L[:, 1] = np.log(L[:, 1])
            L = L[:400]
            attrs = list(range(2, 8)) + list(range(9, 18))
            label = 1
            trIdxs = list(range(200))
            teIdxs = list(range(200, 400))
            Xtr, Ytr, Xte, Yte = partitionData(L, attrs, label, trIdxs, teIdxs)

        elif dataset == "housing":
            L = np.loadtxt("./datasets/housing.txt")
            L = shuffleData(L)
            attrs = [1, 2] + list(range(4, 14))
            label = 0
            trIdxs = list(range(256))
            teIdxs = list(range(256, 506))
            Xtr, Ytr, Xte, Yte = partitionData(L, attrs, label, trIdxs, teIdxs)

        elif dataset == "music":
            L = np.loadtxt("./datasets/music.txt")
            L = shuffleData(L)
            attrs = list(range(1, 91))
            label = 0
            trIdxs = list(range(1000))
            teIdxs = list(range(1000, 2000))
            Xtr, Ytr, Xte, Yte = partitionData(L, attrs, label, trIdxs, teIdxs)
        
        elif dataset == "telemonitoring-total":
            L = np.loadtxt("./datasets/telemonitoring.txt", delimiter=",")
            L = L[L[:, 2] == 0, :]
            L = shuffleData(L)
            attrs = [1, 3, 4] + list(range(6, 22))
            label = 5
            trIdxs = list(range(1000))
            teIdxs = list(range(1000, 1867))
            Xtr, Ytr, Xte, Yte = partitionData(L, attrs, label, trIdxs, teIdxs)

        elif dataset == "forestfires":
            L = np.loadtxt("./datasets/forestfires.txt", delimiter=",")
            L = shuffleData(L)
            L[:, 10] = np.log(L[:, 10] + 1)
            attrs = list(range(6)) + list(range(7, 11))
            label = 6
            trIdxs = list(range(139, 350))
            teIdxs = list(range(350, 517))
            Xtr, Ytr, Xte, Yte = partitionData(L, attrs, label, trIdxs, teIdxs)

        elif dataset == "galaxy":
            L = loadmat("./datasets/lrgReg.mat")
            X = L["X"]
            Y = L["Y"]
            Y /= np.std(Y)
            trIdxs = list(range(2000))
            teIdxs = list(range(2000, 4000))
            Xtr = X[trIdxs, :]
            Xte = X[teIdxs, :]
            Ytr = np.squeeze(Y[trIdxs])
            Yte = np.squeeze(Y[teIdxs])

        elif dataset == "skillcraft":
            L = np.loadtxt("./datasets/skillcraft.txt", delimiter=",")
            L = shuffleData(L)
            attrs = list(range(1, 15)) + list(range(16, 20))
            label = 15
            trIdxs = list(range(1700))
            teIdxs = list(range(1700, 3300))
            Xtr, Ytr, Xte, Yte = partitionData(L, attrs, label, trIdxs, teIdxs)

        elif dataset == "airfoil":
            L = np.loadtxt("./datasets/airfoil.txt")
            L = shuffleData(L)
            attrs = list(range(5))
            label = 5
            trIdxs = list(range(750))
            teIdxs = list(range(750, 1500))
            Xtr, Ytr, Xte, Yte = partitionData(L, attrs, label, trIdxs, teIdxs)
            numAddDims = 36
            Xtr = np.concatenate((Xtr, np.random.randn(Xtr.shape[0], numAddDims)), axis=-1)
            Xte = np.concatenate((Xte, np.random.randn(Xte.shape[0], numAddDims)), axis=-1)

        elif dataset == "CCPP":
            L = loadmat("./datasets/ccpp.mat")
            L = np.concatenate((L["XTrain"], L["YTrain"]), axis=-1)
            L = shuffleData(L)
            attrs = list(range(4))
            label = 4
            trIdxs = list(range(2000))
            teIdxs = list(range(2000, 4000))
            Xtr, Ytr, Xte, Yte = partitionData(L, attrs, label, trIdxs, teIdxs)
            numAddDims = 55
            
            Xtr = np.concatenate((Xtr, np.random.randn(Xtr.shape[0], numAddDims)), axis=-1)
            Xte = np.concatenate((Xte, np.random.randn(Xte.shape[0], numAddDims)), axis=-1)

        elif dataset == "insulin":
            L = loadmat("./datasets/epidata.mat")
            L = np.concatenate((L["insulin_data"], L["snp_data"]), axis=-1)
            L = shuffleData(L)
            trIdxs = list(range(2000))
            teIdxs = list(range(2000, 4000))
            attrs = list(range(1, 51))
            label = 0
            trIdxs = list(range(256))
            teIdxs = list(range(256, 506))
            Xtr, Ytr, Xte, Yte = partitionData(L, attrs, label, trIdxs, teIdxs)

    elif callable(dataset):
        X = make_X(d, m, dist=dist)
        Y = array_map(dataset, X) + noise * np.random.randn(m)
        Xtr = X[:int(m*train_ratio), :]
        Xte = X[int(m*train_ratio):, :]
        Ytr = Y[:int(m*train_ratio)]
        Yte = Y[int(m*train_ratio):]

    return Xtr, Xte, Ytr, Yte

def shuffleData(L):
    m = L.shape[0]
    shuffleOrder = np.random.permutation(m)
    L = L[shuffleOrder, :]
    return L

def partitionData(L, attrs, label, trainIdxs, testIdxs):
    attrs = np.array(attrs)
    trainIdxs = np.array(trainIdxs)
    testIdxs = np.array(testIdxs)
    Xtr = L[trainIdxs[:, np.newaxis], attrs]
    Ytr = L[trainIdxs, label]
    Xte = L[testIdxs[:, np.newaxis], attrs]
    Yte = L[testIdxs, label]
    

    meanXtr = np.mean(Xtr, axis=0)
    stdXtr = np.std(Xtr, axis=0)
    meanYtr = np.mean(Ytr, axis=0)
    stdYtr = np.std(Ytr, axis=0)

    def normalize(x, mean, std):
        if not np.isscalar(mean):
            mean = mean.reshape(1, len(mean))
        if not np.isscalar(std):
            std = std.reshape(1, len(std))
        return (x-mean)/std
    Xtr = normalize(Xtr, meanXtr, stdXtr)
    Xte = normalize(Xte, meanXtr, stdXtr)
    Ytr = normalize(Ytr, meanYtr, stdYtr)
    Yte = normalize(Yte, meanYtr, stdYtr)

    return Xtr, Ytr, Xte, Yte

def array_map(fn, X):
    return np.array(list(map(fn, X)))