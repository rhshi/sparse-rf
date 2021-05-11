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

def fn7(x):
    return (np.sum(x)) ** 2

def fn8(x):
    return np.prod(x)

def fn9(x):
    return x[1] ** 3 + x[3] ** 3

def fn10(x):
    return x[0] * x[1] + x[2] * x[3] + x[1] * x[4]

def fn11(x):
    return x[1] ** 2 + x[2]

def fn12(x):
    return x[0] ** 3 + x[1] ** 2 + x[2]

def fn13(x):
    return x[1] * x[2]

def fn14(x):
    return x[1] * x[2] + x[0] + x[1] ** 2 * x[2]

def fn15(x):
    return x[0] ** 10 + x[1] ** 3