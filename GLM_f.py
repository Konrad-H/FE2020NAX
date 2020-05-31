def GLM(df, regressors, first_year, last_year):

    import numpy as np
    import pandas as pd 

    from sklearn.linear_model import TweedieRegressor
    from sklearn.linear_model import LinearRegression

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import statsmodels.api as sm
    from scipy import stats
    # lasst year is intendeed as 31/12 of last year


    df_reg = df[(df['year']>=first_year) & (df['year']<=last_year)]
    
    last_year = last_year + 1

    firsty_pos = (first_year - 2008)*365
    lasty_pos = (last_year - 2008)*365
    

    

    # %%
    X = regressors[firsty_pos:lasty_pos]
    print('GLM_f')
    print(X)
    y = df.std_demand[firsty_pos:lasty_pos]
    print('y')
    print(y)
    linreg = sm.GLM(y, X, family=sm.families.Gaussian(sm.families.links.identity()))
    linreg_results = linreg.fit()
    print(linreg_results.summary())

    linreg = LinearRegression()
    linreg.fit(X,y)
    #print('Linear')
    #print(linreg)
    #print(linreg.intercept_)
    #print(linreg.coef_)
    #print()

    y_pred = linreg.predict(X)
    residuals = y- y_pred
    sigma = np.sqrt(np.dot(residuals, residuals)/(len(residuals)-1))


    # %%
    # BETA DEL PROF
    beta = [0.385, -0.000016, -0.003, -0.028, 0.136, -0.043, -0.146, -0.120, -0.060]
    

    # %%
    # PLOT GLM
    plt.figure()

    N=len(X)
    x_axis = range(firsty_pos,lasty_pos)

    demand_plt = pd.Series(df_reg.demand,index=x_axis)
    demand_plt.plot()

    M = max(df.log_demand)
    m = min(df.log_demand) 

    demand_pred = y_pred*(M-m) + m
    internew = pd.Series(np.exp(demand_pred),index=x_axis)
    internew.plot()

    plt.show()


    # %%
    # Build matrix for Neural Network
    
    real_y = df.std_demand
    pred_y = linreg.predict(regressors)
    residuals = real_y - pred_y

    #df_temp=pd.DataFrame(residuals)
    #df_temp['drybulb'] = df_reg.drybulb
    #df_temp['dewpnt'] = df_reg.dewpnt
    #df_temp = df_temp.rename(columns={'std_demand':'residuals'})
    #df_temp = df_temp.rename(columns={'residuals'})

    #df_temp['std_demand']=df_reg.std_demand
    #df_temp['log_demand']=df_reg.log_demand
    #N=len(regressors)
    #x_axis = range(firsty_pos,lasty_pos)
    #calendar_var = pd.DataFrame(regressors, index = x_axis)
    #final_df = pd.concat([df_temp, calendar_var],axis=1)
    #final_df.head()

    #final_df.tail()

    # %%
    #final_df.to_csv("train_data.csv")




    # %%
    #%%

    demand_std = pd.Series(y)
    demand_GLM = pd.Series(y_pred+y*0)
    demand_GLM2 = pd.Series(y-residuals)
    # %%
    plt.figure()


    demand_std[1000:1300].plot()
    demand_GLM[1000:1300].plot()
    demand_GLM2[1000:1300].plot()
    plt.show()

    return pred_y, sigma
    # %%
