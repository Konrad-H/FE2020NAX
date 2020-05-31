def pinball(start_date, end_date, y, y_pred, sigma):
    import numpy as np
    from scipy.stats import norm
    from dest_f import destd

    y_pin = np.array(y[(start_date-2008)*365:(end_date+1-2008)*365])

    y_pred_pin_std = y_pred[(start_date-2008)*365:(end_date+1-2008)*365]
    NIP = np.zeros((len(y_pin), 99))

    for ii in range(99):
        alpha = (ii+1)/100
        gauss_quant = norm.ppf(alpha, sigma)
        quant = np.exp((((13.2049-12.4162)*y_pred_pin_std) + 12.4162) + sigma*(13.2049-12.4162)*gauss_quant)
        len(quant)
        for jj in range(len(y_pin)):
            if y_pin[jj] > quant[jj]:
                NIP[jj,ii] = alpha*(y_pin[jj] - quant[jj])
            else:
                NIP[jj,ii] = (1-alpha)*(quant[jj] - y_pin[jj])

    return np.mean(NIP, axis=0)