def ARX(df, first_year, last_year):

    import numpy as np
    import pandas as pd 

    #import pyflux as pf
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import statsmodels.api as sm
    # lasst year is intendeed as 31/12 of last year
    import statsmodels.api as sm
    # lasst year is intendeed as 31/12 of last year
        
    last_year = last_year + 1

    firsty_pos = (first_year - 2008)*365
    lasty_pos = (last_year - 2008)*365

    X = df.iloc[:,[2,3,5,6,7,8,9,10,11,12]]
    y = df.std_demand
    X_arx = X.iloc[firsty_pos:lasty_pos,:]
    y_arx = y.iloc[firsty_pos:lasty_pos]


    # %%

    model3=sm.tsa.ARIMA(endog=y_arx,exog=X_arx,order=[1,0,0])
    results3=model3.fit()

    y_pred = results3.predict(start=0)
    residuals = y_arx - y_pred
    sigma = np.sqrt(np.dot(residuals, residuals)/(len(residuals)-1))
    print(sigma)

        # %%
        # Build matrix for Neural Network
    X_pre = X.iloc[lasty_pos:,:]
    y_pre = y.iloc[lasty_pos:]
    pred_y = results3.predict(start=0, exog=X_pre)
    residuals = y_pre - pred_y


    # %%
    #%%

    demand_std = pd.Series(y_arx)
    demand_ARX = pd.Series(y_pred)
    #demand_ARX2 = pd.Series(y_pred_m[0])
    # %%
    plt.figure()


    demand_std.plot()
    demand_ARX.plot()
    #demand_ARX2.plot()
    plt.show()

    return pred_y, sigma
    # %%
