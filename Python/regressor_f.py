import numpy as np
import pandas as pd 
from standard_and_error_functions import standardize

def regressor(df):

    # This function builds the regressors matrix for GLM 
    #
    # INPUT:
    # df:           Dataframe containing the needed variables to build the regressors
    # OUTPUT:
    # regressors:   Regressors matrix for GLM
  
    N_data = len(df)
    omega = 2*np.pi/365                         # frequence for yearly and semstral trend
    D_weekend = pd.get_dummies(df.day_of_week)  # build a matrix whose vectors are dummy variables for each day of the week
    time_in_days = range(N_data)                # each row of the Dataframe represent a column

    # FIRST DAY IS 1/3/2008, lacking 59 calendar days, as we start to count from the first day in the original dataframe
    time_since_dataset = (2008-2003)*365-59
    t=( np.array(time_in_days)+time_since_dataset )
    
    # reparametrization of time
    time = t/(max(t))

    # Covariates are defined
    regressors =[np.ones(N_data),time,np.sin(omega*t),np.cos(omega*t),np.sin(2*omega*t),np.cos(2*omega*t),D_weekend.Sat, D_weekend.Sun, df.holiday] 
    regressors = np.transpose(regressors)
    
    return regressors
