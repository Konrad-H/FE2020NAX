def ARX(df, first_year, last_year, test_year):

    import numpy as np
    import pandas as pd
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import statsmodels.api as sm

    # last year is intendeed as 31/12 of last year
    last_year = last_year + 1
    test_year = test_year + 1

    firsty_pos = (first_year - first_year)*365
    lasty_pos = (last_year - first_year)*365
    testy_pos = (test_year - first_year)*365

    # %%
    # Build ARX model
    
    X = df.iloc[:,[2,3,5,6,7,8,9,10,11,12]] # take drybulb, dewpnt, all regressors except intercept's one
    y = df.std_demand
    X_train = X.iloc[firsty_pos:lasty_pos,:]
    y_train = y.iloc[firsty_pos:lasty_pos]

    model=sm.tsa.ARIMA(endog=y_train,exog=X_train,order=[1,0,0])
    results=model.fit()

    # Get std. dev. of the model
    y_pred_train, _, _ = results.forecast(steps=len(X_train), exog=X_train, alpha=0.95)
    residuals = y_train - y_pred_train
    sigma = np.sqrt(np.dot(residuals, residuals)/(len(residuals)-1))

    # %%
    # Prediction on test set
    X_test = X.iloc[lasty_pos:testy_pos,:]
    y_pred_test, _, _ = results.forecast(steps=len(X_test), exog=X_test, alpha=0.95)

    # %%
    # Graph on train set
    x_axis = y_train.index
    demand_std = pd.Series(y_train)
    demand_ARX = pd.Series(y_pred_train, index=x_axis)
    
    plt.figure()
    demand_std.plot()
    demand_ARX.plot()
    plt.show()

    return y_pred_test, sigma
    # %%
