def ConfidenceInterval(mu, sigma, alpha):
    #sono zero sicuro che si faccia cosi, domani guardo meglio
    # M e m da salvare global?? o comunque in qualche modo
    M = 13.2049
    m = 12.4162
    
    alfa_l = (1-alpha)/2
    alfa_u = 1-alfa_l
    
    # lower bound
    gauss_quant = norm.ppf(alfa_l, sigma)
    IC_l = np.exp((((M-m)*mu) + m) + sigma*(M-m)*gauss_quant)
    gauss_quant = norm.ppf(alfa_u, sigma)
    IC_u = np.exp((((M-m)*mu) + m) + sigma*(M-m)*gauss_quant)

    return IC_l, IC_u
