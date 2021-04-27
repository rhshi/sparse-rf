import numpy as np

def fn1(x):
    return x[1]

def fn2(x):
    return x[3] ** 2 + x[1] * x[2] + x[0] * x[1] + x[3]

def fn3(x):
    return np.sinc(x[0]) * np.sinc(x[2]) ** 3 + np.sinc(x[1])

def fn4(x):
    return np.exp(-x[1]*x[3]) - 3/2.*(x[0]**2)  - 5.*np.sin(x[2]*x[5]) + (1/2.)*x[2]* x[3]* x[4]

def fn5(x):
    return 5*x[1]/np.power(1+x[4],2) + np.sqrt(x[3] + 1)

def fn6(x):
    return (x[1]*x[3])/(1+x[2]**6)

def fn7(x):
    return np.cos(x[1] + x[4])

def fn8(x):
    return x[1] ** 9 - x[1] ** 8 + 7 * x[0] ** 7 + x[1] ** 6 + 3 * x[3] ** 3 - x[4] + 5 + 0.5 * x[2] ** 15

def fn9(x):
    return np.sum(x ** 3) / np.sum(x ** 1.5)

def fn10(x):
    return x[0] ** 2 + np.sum(x)

def fn11(x):
    s = 0
    for i in range(int(len(x)/2)):
        s += x[i]*x[i+1]*x[i+2]

    return s

def fn12(x):
    s = 0
    for i in range(int(len(x)/2)):
        s += np.cos((x[i]*x[i+1]*x[i+2])**2)

    return s

def fn13(x):
    return np.cos(np.prod(x[:int(len(x/2))]))

def fn14(x):
    return np.prod(np.cos(x[:int(len(x/2))])+1)

def fn15(x):
    return np.cos(np.dot(0.1 * np.ones(len(x)), x))