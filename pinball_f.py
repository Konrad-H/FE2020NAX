def pinball(y, y_test, sigma, M, m):
    import numpy as np
    from scipy.stats import norm
    
    NIP = np.zeros((len(y), 99))

    for ii in range(99):
        alpha = (ii+1)/100
        gauss_quant = norm.ppf(alpha, loc=0, scale=1)
        quant = np.exp((((M-m)*y_test) + m) + sigma*(M-m)*gauss_quant)
        len(quant)
        for jj in range(len(y)):
            if y[jj] > quant[jj]:
                NIP[jj,ii] = alpha*(y[jj] - quant[jj])
            else:
                NIP[jj,ii] = (1-alpha)*(quant[jj] - y[jj])

    return np.mean(NIP, axis=0)