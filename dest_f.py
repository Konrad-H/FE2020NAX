def destd(vector):

    import numpy as np

    M = 13.2049
    m = 12.4162
    log_vec = vector *(M-m) + m*len(vector)
    vec = np.exp(log_vec)
    return vec