import numpy as np

def destd(vector,M,m):
    # function that de-standardizes, from  [0,1] to log to actual demand
    # input:
    #   vector: Nx1 vector to be de-stand
    #   M: max value in log form
    #   m: min value in log form
    log_vec = vector*(M-m) + [m]*len(vector)
    vec = np.exp(log_vec)
    return vec