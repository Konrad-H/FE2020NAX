def destd(vector):

    import numpy as np
    from Run_NAX_Project import M, m

    log_vec = vector*(M-m) + [m]*len(vector)
    vec = np.exp(log_vec)
    return vec