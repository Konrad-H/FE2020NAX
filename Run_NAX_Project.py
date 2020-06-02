# To add a new cell, type '# %%'

# %% Import useful packages
import pandas as pd 
import numpy as np
import os
import sys
from data_mining_f import data_mining, data_standardize
from regressor_f import regressor
from GLM_f import GLM
import time
from dest_f import destd 
import matplotlib.pyplot as plt

# %% Dataset extraction and datamining
tic = time.time()
dataset = data_mining()
toc = time.time()
print(str(toc-tic) + ' sec Elapsed\n')
print(dataset[['demand', 'drybulb', 'dewpnt']].describe())
 
start_date = 2009
end_date   = 2010
start_pos = (start_date -2008)*365
end_pos   = (end_date+1 -2008)*365
 
dataset_plt=dataset[start_pos:end_pos]
plt.figure()
plt.plot(dataset_plt.demand, color = 'red', linewidth=0.5, label='Consumption')
plt.plot(dataset_plt.demand[dataset_plt.day_of_week=='Sun'].index, dataset_plt.demand[dataset_plt.day_of_week=='Sun'].values, 
        linestyle = '', color = 'blue', marker = '.', markersize = 5, label='Sundays')
plt.legend()
plt.show()
# plt.xticks per valori sull'asse x
 
dataset = data_standardize(dataset)
 
global M
M = max(dataset.log_demand)
global m 
m = min(dataset.log_demand)


# %% ex. 1

# %% GLM Model
# define regressors

regressors = regressor(dataset)

# GLM on 2008-2010 time window
start_date = 2009
end_date   = 2011
val_date   = 2012
start_pos = (start_date -2008)*365
end_pos   = (end_date+1 -2008)*365
val_pos   = (val_date+1 -2008)*365
y_GLM_val, y_GLM_train, sigma = GLM(dataset, regressors, start_date, end_date, val_date) #predicted values
y_GLM = np.concatenate([y_GLM_val, y_GLM_train])
residuals = dataset.std_demand[start_pos:val_pos] - y_GLM


# %%
# PLOT GLM

x_axis = range(start_pos, end_pos)
demand_plt = pd.Series(dataset.demand[start_pos:end_pos],index=x_axis)
demand_pred = destd(y_GLM_train)
demand_pred_plt = pd.Series(demand_pred,index=x_axis)

plt.figure()
demand_plt.plot()
demand_pred_plt.plot()
plt.show()

# %% NAX Model
# Needed data stored in a DataFrame

x_axis = range(len(regressors))
calendar_var = pd.DataFrame(regressors, index = x_axis)
calendar_var_NAX = calendar_var[start_pos:val_pos]
temp_data = pd.DataFrame({'std_demand': dataset.std_demand,
                          'residuals': residuals,
                          'drybulb': dataset.drybulb,
                          'dewpnt': dataset.dewpnt})
temp_data_NAX = temp_data[start_pos:val_pos]
dataset_NAX = pd.concat([temp_data_NAX ,calendar_var_NAX],axis=1)

# %% KONRAD ex. 4

# now hyper param.s have been established


# %% ex. 5

start_date = 2009
end_date   = 2011
test_date  = 2012
start_pos = (start_date -2008)*365
end_pos   = (end_date+1 -2008)*365
test_pos  = (test_date+1 -2008)*365
y_GLM_test, y_GLM_train, sigma_GLM = GLM(dataset, regressors, start_date, end_date, test_date) #predicted values
y_GLM = np.concatenate([y_GLM_val, y_GLM_train])
residuals = dataset.std_demand[start_pos:test_pos] - y_GLM

calendar_var_NAX = calendar_var[start_pos:test_pos]
temp_data = pd.DataFrame({'std_demand': dataset.std_demand,
                              'residuals': residuals,
                              'drybulb': dataset.drybulb,
                              'dewpnt': dataset.dewpnt})                             
temp_data_NAX = temp_data[start_pos:test_pos]
dataset_NAX = pd.concat([temp_data_NAX ,calendar_var_NAX],axis=1)

# NAX model returns mu_NAX e sigma_NAX
# y_pred_NAX = y_pred_GLM + mu_NAX
#
# bisogna prendere solo la parte dell'anno 2012

# y_pred_NAX_l, y_pred_NAX_u = ConfidenceInterval(y_pred_NAX, sigma_NAX)

from ConfidenceInterval_f import ConfidenceInterval
y_pred_GLM_l, y_pred_GLM_u = ConfidenceInterval(y_GLM_val, sigma_GLM, 0.95)
 
# Prova backtest
from backtest_f import backtest
y_real = np.array(dataset.demand[end_pos:val_pos])
confidence_levels = np.arange(0.9,1,0.01)
backtested_levels = backtest(y_real, y_GLM_val, confidence_levels, sigma_GLM)
print('backtest')
print(backtested_levels)



# plot da fare


# %% ex. 6

for i in range(4):
    start_date = 2010+i
    end_date   = 2012+i
    test_date  = 2013+i

    #attacco tutto il pezzo che ora scrivo fuori dal for solo per 2010 - 2012 - 2013


start_date = 2009
end_date   = 2011
test_date  = 2012
start_pos = (start_date -2008)*365
end_pos   = (end_date+1 -2008)*365
test_pos  = (test_date+1 -2008)*365
y_GLM_test, y_GLM_train, sigma_GLM = GLM(dataset, regressors, start_date, end_date, test_date) #predicted values
y_GLM = np.concatenate([y_GLM_val, y_GLM_train])

residuals = dataset.std_demand[start_pos:test_pos] - y_GLM


from rmse_f import rmse
from mape_f import mape
print('RMSE_GLM')
print(rmse(dataset.demand[end_pos:test_pos],destd(y_GLM_test)))
print('MAPE_GLM')
print(mape(dataset.demand[end_pos:test_pos],destd(y_GLM_test)))


calendar_var_NAX = calendar_var[start_pos:test_pos]
temp_data = pd.DataFrame({'std_demand': dataset.std_demand,
                              'residuals': residuals,
                              'drybulb': dataset.drybulb,
                              'dewpnt': dataset.dewpnt})                             
temp_data_NAX = temp_data[start_pos:test_pos]
dataset_NAX = pd.concat([temp_data_NAX ,calendar_var_NAX],axis=1)

# NAX model returns mu_NAX e sigma_NAX
# y_NAX_test = y_GLM_test + mu_NAX_test

# %% ARX Model
from ARX_f import ARX

y_ARX_test, sigma_ARX = ARX(dataset_NAX, start_date, end_date, test_date) #predicted values

print('RMSE_ARX')
print(rmse(dataset.demand[end_pos:val_pos],destd(y_ARX_test)))
print('MAPE_ARX')
print(mape(dataset.demand[end_pos:test_pos],destd(y_ARX_test)))
# %% pinball
from pinball_f import pinball
from backtest_f import backtest

y = np.array(dataset.demand[end_pos:test_pos])  # demand, non std_demand!!!!!
y_ARX_test = np.array(y_ARX_test)

pinball_values_GLM = pinball(y, y_GLM_test, sigma_GLM)   # y_pred = output of GLM (prediction of std(log_demand))
#pinball_values_NAX = pinball(y, y_NAX_test, sigma_NAX)
pinball_values_ARX = pinball(y, y_ARX_test, sigma_ARX)

pinplot_GLM = pd.Series(pinball_values_GLM)
#pinplot_NAX = pd.Series(pinball_values_NAX)
pinplot_ARX = pd.Series(pinball_values_ARX)

# pinball graph
plt.figure()
pinplot_GLM.plot()
#pinplot_NAX.plot()
pinplot_ARX.plot()
plt.show()

# %% backtest
from backtest_f import backtest

print(sigma_GLM)
print(sigma_ARX)

print('backtest')
confidence_levels = np.arange(0.9,1,0.01)
backtested_levels_GLM, LR_Unc_GLM, LR_Cov_GLM = backtest(y, y_GLM_test, confidence_levels, sigma_GLM)
#backtested_levels_NAX, LR_Unc_NAX, LR_Cov_NAX = backtest(y, y_NAX_val, confidence_levels, sigma_NAX)
backtested_levels_ARX, LR_Unc_ARX, LR_Cov_ARX = backtest(y, y_ARX_test, confidence_levels, sigma_ARX)

print('LR_GLM')
print(LR_Unc_GLM, LR_Cov_GLM)
#print('LR_NAX')
#print(LR_Unc_NAX, LR_Cov_NAX)
print('LR_ARX')
print(LR_Unc_ARX, LR_Cov_ARX)

backplot_GLM = pd.Series(backtested_levels_GLM)
#backplot_NAX = pd.Series(backtested_levels_NAX)
backplot_ARX = pd.Series(backtested_levels_ARX)
confplot     = pd.Series(confidence_levels)

plt.figure()
backplot_GLM.plot()
#backplot_NAX.plot()
backplot_ARX.plot()
confplot.plot()
plt.show()

a=b



# %%
