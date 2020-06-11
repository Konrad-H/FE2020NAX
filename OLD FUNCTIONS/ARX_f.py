import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import statsmodels.api as sm

def ARX(df, first_year, last_year, test_year):

    # This function implements an ARX model
    #
    # INPUTS:
    # df:           DataFrame containing varible std_demand, i.e. response to be predicted, 
    #               and the exogenous regressors: calendar variables and weather conditions
    # first_year:   Year from where the train set starts at 1/1
    # last_year:    Year in which the train set ends at 31/12
    # test_year:    Year of the test_set from 1/1 to 31/12
    #
    # OUTPUTS:
    # y_pred_test:  Predicted response on test set
    # sigma:        Standard Deviation of the fitted model

    # last year is intendeed as 31/12 of last year
    last_year = last_year + 1
    test_year = test_year + 1

    firsty_pos = (first_year - first_year)*365
    lasty_pos = (last_year - first_year)*365
    testy_pos = (test_year - first_year)*365

    # %%
    # Build ARX model
    # Take exogenous variables on the train set
    X = df.iloc[:,[2,3,5,6,7,8,9,10,11,12]] # take drybulb, dewpnt and the calendar regressors
    y = df.std_demand
    X_train = X.iloc[firsty_pos:lasty_pos,:] # take the years of train set 
    y_train = y.iloc[firsty_pos:lasty_pos]   # take the years of train set 

    # model is initialized and fitted on train set
    model=sm.tsa.ARIMA(endog=y_train,exog=X_train,order=[1,0,0]) # AR: 1, I: 0, MA: 0
    results=model.fit()

    # Get standard deviation of the model
    y_pred_train, _, _ = results.forecast(steps=len(X_train), exog=X_train, alpha=0.95)
    residuals = y_train - y_pred_train
    sigma = np.sqrt(np.dot(residuals, residuals)/(len(residuals)-1))

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
