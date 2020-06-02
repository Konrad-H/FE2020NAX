def backtest(y_real, y_pred, confidence_levels, sigma):
    import numpy as np
    from ConfidenceInterval_f import ConfidenceInterval

    backtested_levels = np.zeros((len(confidence_levels)))
    N = len(y_real)

    for cl in range(len(confidence_levels)):
        IC_l, IC_u = ConfidenceInterval(y_pred, sigma, confidence_levels[cl])
        backtested_levels[cl] = sum(((y_real>=IC_l) & (y_real<=IC_u)))
        if cl==5:
            exception_vec = [1]*len(y_real) - ((y_real>=IC_l) & (y_real<=IC_u))
    
    # Unconditional Covaraage at 95% confidence level
    backtested_95 = backtested_levels[5]
    exceptions = N - backtested_95
    alpha = 1-confidence_levels[5]
    alpha_hat = exceptions/N

    LR_Uncond = -2*np.log(((alpha/alpha_hat)**exceptions)*(((1-alpha)/(1-alpha_hat))**backtested_95))

    # Conditional Covarage at 95% confidence level

    N_mat = np.zeros((2,2)) # N_00, N_01
                            # N_10, N_11
    flag=0
    for ii in range(len(exception_vec)):
        if exception_vec[ii]==1:
            if flag==1:
                N_mat[1,1] += 1
            else:
                N_mat[0,1] += 1
                flag=1
        else:
            if flag==1:
                N_mat[1,0] += 1
                flag=0
            else:
                N_mat[0,0] += 1

    alpha_01 = N_mat[0,1]/(sum(N_mat[0,:]))
    alpha_11 = N_mat[1,1]/(sum(N_mat[1,:]))

    num = (alpha**exceptions)*((1-alpha)**backtested_95)
    den = (1-alpha_01)**N_mat[0,0]*alpha_01**N_mat[0,1]*(1-alpha_11)**N_mat[1,0]*alpha_11**N_mat[1,1]

    LR_Cond = -2*np.log(num/den)

    return backtested_levels/N, LR_Uncond, LR_Cond