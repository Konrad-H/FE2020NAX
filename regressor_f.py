def regressor(df):

    import numpy as np
    import pandas as pd 

    # %% Build the regressors matrix
  
    N_data = len(df)
    print()
    print(N_data)
    omega = 2*np.pi/365
    D_weekend = pd.get_dummies(df.day_of_week)
    time_in_days = range(N_data)

    # FIRST DAY IS march 1st, lacking 59 calendar days
    time_since_dataset = (2008-2003)*365-59

    t=( np.array(time_in_days)+time_since_dataset )
    # covariates
    regressors =[np.ones(N_data),t,np.sin(omega*t),np.cos(omega*t),np.sin(2*omega*t),np.cos(2*omega*t),D_weekend.Sat, D_weekend.Sun, df.holiday] 
    regressors = np.transpose(regressors)
    
    return regressors
