def ConfidenceInterval(mu, sigma, confidence_level):

    import numpy as np
    from scipy.stats import norm
    from Run_NAX_Project import M, m
    from dest_f import destd

    alpha_l = (1-confidence_level)/2
    alpha_u = 1-alpha_l

    # lower bound
    gauss_quant_l = norm.ppf(alpha_l, loc=0, scale = 1)
    IC_l = np.exp((((M-m)*mu) + m) + sigma*(M-m)*gauss_quant_l)
    gauss_quant_u = norm.ppf(alpha_u, loc=0, scale = 1)
    IC_u = np.exp((((M-m)*mu) + m) + gauss_quant_u*sigma*(M-m))
    
    return IC_l, IC_u
