import numpy as np

def fn1(x):
    return np.sum(x[:-1]) + np.exp(-x[-1])

def fn2(x):
    return np.cos(x[0]) + np.sin(x[1])

def fn3(x):
    return (2*x[0]-1)*(2*x[1]-1)

def fn4(x):
    return (2*x[0]-1)*(2*x[1]-1) + (2*x[0]-1)*(2*x[2]-1) + (2*x[2]-1)*(2*x[1]-1)

def fn5(x):
    return np.sinc(x[0]) * np.sinc(x[2]) ** 3 + np.sinc(x[1])

def fn6(x):
    return np.sin(x[0]) + 7 * np.sin(x[1]) ** 2 + 0.1 * x[2] ** 4 * np.sin(x[0])

def fn7(x):
    return np.cos(x[0]) * x[2] + x[1]**2 * x[3] + np.sum(x[2:])

def fn8(x, r1, r2, r3, r4):
    return np.cos(r1*x[0]+r2*x[1]) + 3 * np.cos(r3*x[2]+r4*x[3])
    
def fn9(x, r):
    return np.cos(x[0]) + r*np.cos(r*x[1])

