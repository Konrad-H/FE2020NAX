import numpy as np
from scipy.stats import norm


def ConfidenceInterval(mu, sigma, confidence_level, M, m):

    # This function computes lower and upper bound of a confidence interval at confidence_level for each element y_t
    # of a vector y, with standardized log(y_t) distributed as N(mu_t, sigma_t^2)
    # 
    # INPUTS:
    # mu:       vector of mean values
    # sigma:    vector of standard deviations
    # confidence_level
    # M:        maximum of the log_demand observed over the years: 2008 - 2016
    # m:        minimum of the log_demand observed over the years: 2008 - 2016
    #
    # OUTPUTS:
    # IC_l:     vector of lower bounds
    # IC_u:     vector of upper bounds

    alpha_l = (1-confidence_level)/2
    alpha_u = 1-alpha_l

    # lower bound
    gauss_quant_l = norm.ppf(alpha_l, loc=0, scale=1)
    IC_l = np.exp((((M-m)*mu) + m) + sigma*(M-m)*gauss_quant_l)

    # upper bound
    gauss_quant_u = norm.ppf(alpha_u, loc=0, scale=1)
    IC_u = np.exp((((M-m)*mu) + m) + gauss_quant_u*sigma*(M-m))
    
    return IC_l, IC_u


def pinball(y, y_test, sigma, M, m):

    # This function computes Pinball Loss for every quantile
    #
    # INPUTS:
    # y:        vector of the realised demand on the test set
    # y_test:   vector of mean values of the predicted standardized log_demand
    # sigma:    vector of standard deviations of the predicted standardized log_demand
    # M:        maximum of the log_demand observed over the years: 2008 - 2016
    # m:        minimum of the log_demand observed over the years: 2008 - 2016
    #
    # OUTPUT:
    # pinball_values:   vector of Pinball Loss for every quantile
    
    NIP = np.zeros((len(y), 99))

    for ii in range(99):
        alpha = (ii+1)/100
        # compute quantiles of predicted demand, knowing that each element of the predicted 
        # standardized log_demand is distributed as N(y_test_t, sigma_t^2)
        gauss_quant = norm.ppf(alpha, loc=0, scale=1)
        quant = np.exp((((M-m)*y_test) + m) + sigma*(M-m)*gauss_quant)
        
        # fill the NIP matrix
        for jj in range(len(y)):
            if y[jj] > quant[jj]:
                NIP[jj,ii] = alpha*(y[jj] - quant[jj])
            else:
                NIP[jj,ii] = (1-alpha)*(quant[jj] - y[jj])
    
    # take the mean across the test set
    pinball_values = np.mean(NIP, axis=0)

    return pinball_values


def backtest(y_real, y_pred, confidence_levels, sigma, M, m):

    # This function provides backtested_levels, Unconditional Coverage Likelihood Ratio and
    # Conditional Coverage Likelihood Ratio
    #
    # INPUTS
    # y_real:   vector of the realised demand on the test set
    # y_pred:   vector of mean values of the predicted standardized log_demand
    # confidence_levels:    vector of confidence levels
    # sigma:    vector of standard deviations of the predicted standardized log_demand
    # M:        maximum of the log_demand observed over the years: 2008 - 2016
    # m:        minimum of the log_demand observed over the years: 2008 - 2016
    #
    # OUTPUTS:
    # backtested_levels/N:  fraction of realised demand falling inside a confidence interval
    # LR_Uncond:    Unconditional Coverage Likelihood Ratio
    # LR_Cond:      Conditional Coverage Likelihood Ratio
    
    backtested_levels = np.zeros((len(confidence_levels)))
    N = len(y_real)

    # for every confidence level compute the corresponding confidence interval and count the realised
    # demands falling inside the confidence interval
    for cl in range(len(confidence_levels)):
        IC_l, IC_u = ConfidenceInterval(y_pred, sigma, confidence_levels[cl], M, m)
        backtested_levels[cl] = sum(((y_real>=IC_l) & (y_real<=IC_u)))
        
        if cl==5:   # 95% confidence level
            # exception_vec_t is 0 if y_real_t falls inside the corresponding 95% confidence interval, 1 otherwise
            # needed for Conditional Coverage Likelihood Ratio
            exception_vec = [1]*len(y_real) - ((y_real>=IC_l) & (y_real<=IC_u))
 
    # Unconditional Coverage at 95% confidence level
    backtested_95 = backtested_levels[5]
    exceptions = N - backtested_95
    alpha = 1-confidence_levels[5]
    alpha_hat = exceptions/N

    LR_Uncond = -2*np.log(((alpha/alpha_hat)**exceptions)*(((1-alpha)/(1-alpha_hat))**backtested_95))

    # Conditional Coverage at 95% confidence level
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