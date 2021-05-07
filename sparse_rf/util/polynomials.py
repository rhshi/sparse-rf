import numpy as np

def fn1(x):
    return x[1]

def fn2(x):
    return np.sum(x)

def fn3(x):
    return x[3] ** 2 + x[1] * x[2] + x[0] * x[1] + x[3]

def fn4(x):
    return x[1] ** 2

def fn5(x):
    return x[0] ** 2 + np.sum(x)

def fn6(x):
    return x[1] - 2 * x[3] + 1