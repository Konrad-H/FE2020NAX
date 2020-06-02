def GLM(df, regressors, first_year, last_year, test_year):

    import numpy as np
    import pandas as pd 
    
    from sklearn.linear_model import LinearRegression

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import statsmodels.api as sm
    from scipy import stats
    from dest_f import destd
    
    #df_reg = df[(df['year']>=first_year) & (df['year']<=last_year)]
    
    # last year and test_year are intendeed as 31/12 of that year
    last_year = last_year + 1
    test_year = test_year + 1

    firsty_pos = (first_year - 2008)*365
    lasty_pos = (last_year - 2008)*365
    testy_pos = (test_year - 2008)*365
    
    # %%
    # Build GLM model

    X_train = regressors[firsty_pos:lasty_pos]
    y_train = df.std_demand[firsty_pos:lasty_pos]

    # method 1 - summary bella
    linreg = sm.GLM(y_train, X_train, family=sm.families.Gaussian(sm.families.links.identity()))
    linreg_results = linreg.fit()
    print(linreg_results.summary())

    # method 2 - più leggero
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    print('Linear')
    print(linreg.intercept_)
    print(linreg.coef_)
    print()

    # Get std. dev. of the model
    y_pred_train = linreg.predict(X_train)
    residuals = y_train - y_pred_train
    sigma = np.sqrt(np.dot(residuals, residuals)/(len(residuals)-1))

    # %%
    # Prediction on test set
    y_pred_test = linreg.predict(regressors[lasty_pos:testy_pos])

    return y_pred_test, y_pred_train, sigma
    # %%
