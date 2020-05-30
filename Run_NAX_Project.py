# To add a new cell, type '# %%'

# %% Import useful packages
import pandas as pd 
import numpy as np
import os
import sys
from data_mining_f import data_mining
from regressor_f import regressor
from GLM_f import GLM


# %% Dataset extraction and datamining
dataset = data_mining() 
print(dataset)


# %% GLM Model
# define regressors
regressors = regressor(dataset)
print(regressors)

# GLM on 2008-2010 time window      # non mi funziona e non so perchèèèèèè   ora penso che vada!!
start_date = 2008
end_date   = 2010
y_pred, sigma = GLM(dataset, regressors, start_date, end_date) #predicted values
residuals = dataset.std_demand - y_pred
y = dataset.demand  # demand, non std_demand!!!!!


# %% pinball
from scipy.stats import lognorm
import matplotlib.pyplot as plt
from dest_f import destd



start_date = 2011
end_date = 2011
y_pin = y[(start_date-2008)*365:(end_date+1-2008)*365]
y_pin = np.array(y_pin)
NIP = np.zeros((len(y_pin), 99))
len(y_pin)
y_pred_pin_std = y_pred[(start_date-2008)*365:(end_date+1-2008)*365]
y_pred_pin = destd(y_pred_pin_std)


for ii in range(99):
    alpha = (ii+1)/100
    gauss_quant = lognorm.ppf(alpha, sigma)
    quant = y_pred_pin + gauss_quant*sigma
    len(quant)
    for jj in range(len(y_pin)):
        if y_pin[jj] > quant[jj]:
            NIP[jj,ii] = alpha*(y_pin[jj] - quant[jj])
        else:
            NIP[jj,ii] = (1-alpha)*(quant[jj] - y_pin[jj])

mean_NIP = np.mean(NIP, axis=0)
print(mean_NIP.shape)

n = pd.Series(mean_NIP)
plt.figure()
n.plot()
plt.show()



# %% NAX Model
# Needed data stored in a DataFrame

x_axis = range(len(regressors))
calendar_var = pd.DataFrame(regressors, index = x_axis)
temp_data = pd.DataFrame({'residuals': residuals,
                          'drybulb': dataset.drybulb,
                          'dewpnt': dataset.dewpnt})
dataset_NAX = pd.concat([temp_data,calendar_var],axis=1)
print (dataset_NAX)

# %% NAX Hyperparam. Calibration with Calibration on: 2008 - 2010, Validation on 2011


# %% NAX Calibration 

start_cal = 2008
end_cal   = 2010
y_pred = GLM(dataset, regressors, start_date, end_date) #predicted values

test_date = 2012






# %%
