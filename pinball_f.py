def pinball():
    from scipy.stats import norm
    import matplotlib.pyplot as plt
    from dest_f import destd

    start_date = 2011
    end_date = 2011
    y_pin = y[(start_date-2008)*365:(end_date+1-2008)*365]
    y_pin = np.array(y_pin)
    NIP = np.zeros((len(y_pin), 99))
    len(y_pin)
    y_pred_pin_std = y_pred[(start_date-2008)*365:(end_date+1-2008)*365]
    y_pred_pin = destd(y_pred_pin_std)


    for ii in range(99):
        alpha = (ii+1)/100
        gauss_quant = norm.ppf(alpha)
        quant = y_pred_pin + gauss_quant*sigma
        len(quant)
        for jj in range(len(y_pin)):
            if y_pin[jj] > quant[jj]:
                NIP[jj,ii] = alpha*(y_pin[jj] - quant[jj])
            else:
                NIP[jj,ii] = (1-alpha)*(quant[jj] - y_pin[jj])

    return mean_NIP = np.mean(NIP, axis=0)
