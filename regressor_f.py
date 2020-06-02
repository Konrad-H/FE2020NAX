def regressor(df):

    import numpy as np
    import pandas as pd 
    from standardize_f import standardize 
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
    time = t/max(t) # reparametrization of time
    # covariates
    regressors =[np.ones(N_data),time,np.sin(omega*t),np.cos(omega*t),np.sin(2*omega*t),np.cos(2*omega*t),np.array(D_weekend.Sat), np.array(D_weekend.Sun), np.array(df.holiday)] 
    # cose a caso
   
    regressors = np.transpose(regressors)
    
    return regressors
