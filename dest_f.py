import numpy as np

def destd(vector,M,m):
    log_vec = vector*(M-m) + [m]*len(vector)
    vec = np.exp(log_vec)
    return vec