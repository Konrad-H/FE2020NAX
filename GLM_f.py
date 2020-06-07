import numpy as np
import pandas as pd 
import statsmodels.api as sm  


def GLM(df, regressors, first_year, last_year, test_year):

    # This function implements the GLM
    #
    # INPUTS:
    # df:           DataFrame containing varible std_demand: response to be predicted
    # regressors:   DataFrame containing the regressors of the GLM model to be fitted
    # first_year:   Year from where the train set starts at 1/1
    # last_year:    Year in which the train set ends at 31/12
    # test_year:    Year of the test_set from 1/1 to 31/12
    #
    # OUTPUTS:
    # y_pred_test:  Predicted response on test set
    # y_pred_train: Predicted response on train set
    # sigma:        Standard Deviation of the fitted model
    
    # last year and test_year are intendeed as 31/12 of that year
    last_year = last_year + 1
    test_year = test_year + 1

    firsty_pos = (first_year - 2008)*365
    lasty_pos = (last_year - 2008)*365
    testy_pos = (test_year - 2008)*365
    
    # Build GLM model
    # Take regressors on the wanted time window
    X_train = regressors[firsty_pos:lasty_pos]
    y_train = df.std_demand[firsty_pos:lasty_pos]

    # Ordinary Least Square model is fitted
    linreg = sm.OLS(y_train,X_train).fit()
    print(linreg.summary()) # returns summury of the fitted model

    # Get standard deviation of the model
    y_pred_train = linreg.predict(X_train)
    residuals = y_train - y_pred_train
    sigma = np.sqrt(np.dot(residuals, residuals)/(len(residuals)-1))

    # Prediction of the response on test set
    y_pred_test = linreg.predict(regressors[lasty_pos:testy_pos])

    return y_pred_test, y_pred_train, sigma
