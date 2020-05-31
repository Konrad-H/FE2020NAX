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
dataset = data_mining("gefcom.csv") 
print (dataset)


# %% GLM Model
# define regressors
regressors = regressor(dataset)
print(regressors)

# GLM on 2008-2010 time window      # non mi funziona e non so perchèèèèèè   ora penso che vada!!
start_date = 2009
end_date   = 2011
y_pred , X, y= GLM(dataset, regressors, start_date, end_date) #predicted values
print('GLM_f')
print(X)
print('y')
print(y)
residuals = dataset.std_demand - y_pred

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
