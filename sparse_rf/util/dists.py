import numpy as np
from scipy.stats import cauchy

def sphere(shape):
    x = np.random.normal(size=shape)
    return (x.T / np.linalg.norm(x.T, axis=0)).T

def normal(shape, mean=0, stdev=1):
    return stdev * np.random.normal(size=shape) + mean

def uniform(shape, low=-1, high=1):
    return np.random.uniform(size=shape)*(high-low) + low

def cauchy(shape, loc=0, scale=1):
    return cauchy.rvs(loc=loc, scale=scale, size=shape)