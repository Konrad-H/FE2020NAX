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
last_year = 2016


df = df[(df['year']>=first_year) & (df['year']<=last_year)]
N_data = len(df)

log_consumption = np.log(df.demand)
omega = 2*np.pi/365
D_weekend = pd.get_dummies(df.day_of_week)
time_in_days = range(N_data)

# FIRST DAY IS march 1st, lacking 59 calendar days  6 years from 2003 to before 2009 6*365-59 number of days until 2009
time_since_dataset = (first_year-2003)*365-59

t=( np.array(time_in_days)+time_since_dataset )
# covariates
regressors =[np.ones(N_data),t,np.sin(omega*t),np.cos(omega*t),np.sin(2*omega*t),np.cos(2*omega*t),D_weekend.Sat, D_weekend.Sun, df.holiday] 
regressors = np.transpose(regressors)

#print(X)

# %%
Xnew = regressors[:(365*3)]
Y= df.demand[:(365*3)]

gauss_log = sm.GLM(df.demand[:(365*3)], Xnew, family=sm.families.Gaussian(sm.families.links.log))
gauss_log_results = gauss_log.fit()
param_new=gauss_log_results.params


y_pred = gauss_log_results.predict(Xnew )
residuals = y_pred- Y




# %%
# BETA DEL PROF
beta = [0.385, -0.000016, -0.003, -0.028, 0.136, -0.043, -0.146, -0.120, -0.060]


# %%
# PLOT GLM
plt.figure()

df.demand.plot()

N=len(Xnew)
x_axis = range(366,366+N)

internew = pd.Series(np.exp(np.array(np.dot(Xnew,param_new))),index=x_axis)
internew.plot()

plt.show()

# %%
df_temp=pd.DataFrame(residuals)
df_temp['drybulb'] = df.drybulb
df_temp['dewpnt'] = df.dewpnt
df_temp = df_temp.rename(columns={'demand':'residuals'})
calendar_var = pd.DataFrame(Xnew, index = x_axis)

final_df = pd.concat([df_temp, calendar_var],axis=1)
final_df.head()
# %%
final_df.to_csv("train_data.csv")



# %%
