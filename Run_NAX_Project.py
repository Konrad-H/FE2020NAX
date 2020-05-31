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

# GLM on 2008-2010 time window      
start_date = 2008
end_date   = 2010
y_pred, sigma = GLM(dataset, regressors, start_date, end_date) #predicted values
residuals = dataset.std_demand - y_pred

# Confidence interval provati con GLM su anno 2011
from ConfidenceInterval_f import ConfidenceInterval
start_2011_pos = (2011-2008)*365
end_2011_pos   = (2011+1-2008)*365
y_pred_2011 = y_pred[start_2011_pos:end_2011_pos]
y_pred_GLM_l, y_pred_NAX_u = ConfidenceInterval(y_pred_2011, sigma)


# %% NAX Model
# Needed data stored in a DataFrame

x_axis = range(len(regressors))
calendar_var = pd.DataFrame(regressors, index = x_axis)
temp_data = pd.DataFrame({'std_demand': dataset.std_demand,
                              'residuals': residuals,
                              'drybulb': dataset.drybulb,
                              'dewpnt': dataset.dewpnt})                             
dataset_NAX = pd.concat([temp_data,calendar_var],axis=1)

# %% KONRAD ex. 4

# now hyper param.s have been established


# %% ex. 5

start_date = 2009
end_date   = 2011
test_date  = 2012
y_pred_GLM, sigma_GLM = GLM(dataset, regressors, start_date, end_date) #predicted values
residuals = dataset.std_demand - y_pred

x_axis = range(len(regressors))
calendar_var = pd.DataFrame(regressors, index = x_axis)
temp_data = pd.DataFrame({'std_demand': dataset.std_demand,
                              'residuals': residuals,
                              'drybulb': dataset.drybulb,
                              'dewpnt': dataset.dewpnt})                             
dataset_NAX = pd.concat([temp_data,calendar_var],axis=1)

# NAX model returns mu_NAX e sigma_NAX
# y_pred_NAX = y_pred_GLM + mu_NAX
#
# bisogna prendere solo la parte dell'anno 2012

# y_pred_NAX_l, y_pred_NAX_u = ConfidenceInterval(y_pred_NAX, sigma_NAX)



# %% ex. 6

for i in range(4):
    start_date = 2010+i
    end_date   = 2012+i
    test_date  = 2013+i

    #attacco tutto il pezzo che ora scrivo fuori dal for solo per 2009 - 2011 - 2012


start_date = 2010
end_date   = 2012
test_date  = 2013
y_pred_GLM, sigma_GLM = GLM(dataset, regressors, start_date, end_date) #predicted values
residuals = dataset.std_demand - y_pred

x_axis = range(len(regressors))
calendar_var = pd.DataFrame(regressors, index = x_axis)
temp_data = pd.DataFrame({'std_demand': dataset.std_demand,
                              'residuals': residuals,
                              'drybulb': dataset.drybulb,
                              'dewpnt': dataset.dewpnt})                             
dataset_NAX = pd.concat([temp_data,calendar_var],axis=1)

# NAX model returns mu_NAX e sigma_NAX
# y_pred_NAX = y_pred_GLM + mu_NAX

# %% ARX Model
from ARX_f import ARX

start_date = 2009
end_date   = 2011

y_pred_ARX, sigma_ARX = ARX(dataset_NAX, start_date, end_date) #predicted values

# %% pinball
from pinball_f import pinball

start_date = 2011
end_date = 2011
y = dataset.demand  # demand, non std_demand!!!!!

pinball_values_GLM = pinball(start_date, end_date, y, y_pred_GLM, sigma_GLM)   # y_pred = output of GLM (prediction of std(log_demand))
pinball_values_NAX = pinball(start_date, end_date, y, y_pred_NAX, sigma_NAX)
pinball_values_ARX = pinball(start_date, end_date, y, y_pred_ARX, sigma_ARX)

pinplot_GLM = pd.Series(pinball_values_GLM)
pinplot_NAX = pd.Series(pinball_values_NAX)
pinplot_ARX = pd.Series(pinball_values_ARX)

# pinball graph
import matplotlib.pyplot as plt
plt.figure()
pinplot_GLM.plot()
pinplot_NAX.plot()
pinplot_ARX.plot()
plt.show()

# %% backtest



# %%
