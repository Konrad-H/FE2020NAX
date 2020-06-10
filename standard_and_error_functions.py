import numpy as np

def standardize(vector):

    # This function standardizes any vector, s.t. all of his values are mapped in [0,1]
    #
    # INPUT:
    # vector:   vector to be standardized in the interval [0,1]
    #
    # OUTPUT:
    # std_vec:  vector standardized in the interval [0,1]

    M_v = max(vector)
    m_v = min(vector)
    std_vec = (vector - [m_v]*len(vector))/(M_v - m_v)
    return std_vec


def destd(vector,M,m):

    # This function destandardizes any vector of values in [0,1] back in the domain of the demand
    #
    # INPUT:
    # vector:   standardized vector to be destandardized in the demand domain
    # M:        maximum of the log_demand observed over the years: 2008 - 2016
    # m:        minimum of the log_demand observed over the years: 2008 - 2016
    #
    # OUTPUT:
    # vec:      vector destandardized in the domain of the demand

    log_vec = vector*(M-m) + [m]*len(vector)
    vec = np.exp(log_vec)
    return vec


def rmse(predictions, targets):

    # This function computes the Root Mean Squared Error (RMSE)
    #
    # INPUT:
    # predictions:  predicted values
    # targets:      real values
    #
    # OUTPUT: 
    # RMSE:         Root Mean Squared Error

    return np.sqrt(((predictions - targets) ** 2).mean())


def mape(predictions, targets):

    # This function computes the Mean Absolute Percentage Error (MAPE)
    #
    # INPUT:
    # predictions:  predicted values
    # targets:      real values  
    # 
    # OUTPUT: 
    # MAPE:         Mean Absolute Percentage Error

    return np.mean(np.abs((targets-predictions)/targets))