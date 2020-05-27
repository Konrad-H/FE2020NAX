# GLM 
#
# %% IMPORT 
import numpy as np
import pandas as pd 

from sklearn.linear_model import TweedieRegressor
from sklearn.linear_model import LinearRegression

import matplotlib as mpl
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats


# %% Build the regressors matrix


df = pd.read_csv("gefcom_standard.csv") 

first_year = 2009
last_year = 2011


df_reg = df[(df['year']>=first_year) & (df['year']<=last_year)]
print(df_reg)
N_data = len(df_reg)
omega = 2*np.pi/365
D_weekend = pd.get_dummies(df_reg.day_of_week)
time_in_days = range(N_data)

# FIRST DAY IS march 1st, lacking 59 calendar days  6 years from 2003 to before 2009 6*365-59 number of days until 2009
time_since_dataset = (first_year-2003)*365-59

t=( np.array(time_in_days)+time_since_dataset )
# covariates
regressors =[np.ones(N_data),t,np.sin(omega*t),np.cos(omega*t),np.sin(2*omega*t),np.cos(2*omega*t),D_weekend.Sat, D_weekend.Sun, df_reg.holiday] 
regressors = np.transpose(regressors)


# %%
X = regressors[:(365*3)]
y = df_reg.std_demand[:(365*3)]

linreg = sm.GLM(y, X, family=sm.families.Gaussian(sm.families.links.identity()))
linreg_results = linreg.fit()
print(linreg_results.summary())

linreg = LinearRegression()
linreg.fit(X,y)
print('Linear')
print(linreg)
print(linreg.intercept_)
print(linreg.coef_)
print()

y_pred = linreg.predict(X)
residuals = y - y_pred


# %%
# BETA DEL PROF
beta = [0.385, -0.000016, -0.003, -0.028, 0.136, -0.043, -0.146, -0.120, -0.060]


# %%
# PLOT GLM
plt.figure()

N=len(X)
x_axis = range(365,365+N)

demand_plt = pd.Series(df_reg.demand,index=x_axis)
demand_plt.plot()

M = max(df.log_demand)
m = min(df.log_demand)
demand_pred = y_pred*(M-m) + m
internew = pd.Series(np.exp(demand_pred),index=x_axis)
internew.plot()

plt.show()

# %%
df_temp=pd.DataFrame(residuals)
df_temp['drybulb'] = df.drybulb
df_temp['dewpnt'] = df.dewpnt
df_temp = df_temp.rename(columns={'demand':'residuals'})
N=len(regressors)
x_axis = range(365,365+N)
calendar_var = pd.DataFrame(regressors, index = x_axis)
final_df = pd.concat([df_temp, calendar_var],axis=1)
final_df.tail()
print(final_df)
# %%
final_df.to_csv("train_data.csv")



# %%
