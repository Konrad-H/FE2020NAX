# To add a new cell, type '# %%'

# %% Import useful packages
import pandas as pd 
import numpy as np
import os
import sys
from data_mining_f import data_mining
from regressor_f import regressor
from GLM_f import GLM
import time


# %% Dataset extraction and datamining
tic = time.time()
dataset = data_mining()
toc = time.time()
print(str(toc-tic) + ' sec Elapsed')
dataset


# %% GLM Model
# define regressors
regressors = regressor(dataset)
print(regressors)

# GLM on 2008-2010 time window      # non mi funziona e non so perchèèèèèè   ora penso che vada!!
start_date = 2008
end_date   = 2010
y_pred, sigma = GLM(dataset, regressors, start_date, end_date) #predicted values
residuals = dataset.std_demand - y_pred


# %% pinball
from pinball_f import pinball

start_date = 2011
end_date = 2011
y = dataset.demand  # demand, non std_demand!!!!!

pinball_values = pinball(start_date, end_date, y, y_pred, sigma)   # y_pred = output of GLM (prediction of std(log_demand))

import matplotlib.pyplot as plt
n = pd.Series(pinball_values)
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
