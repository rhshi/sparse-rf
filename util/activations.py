import numpy as np

def relu(x):
    return np.maximum(0, x)

def fourier(x):
    return np.concatenate((np.cos(x), np.sin(x)), axis=-1)