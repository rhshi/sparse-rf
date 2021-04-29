from scipy.special import comb as comb_

def comb(n, k):
    return int(comb_(n, k))