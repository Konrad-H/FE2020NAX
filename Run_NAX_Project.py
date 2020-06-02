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
dataset = data_mining("gefcom.csv")
toc = time.time()
print(str(toc-tic) + ' sec Elapsed\n')
print(dataset[['demand', 'drybulb', 'dewpnt']].describe())
 
# %%

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
y_GLM = np.concatenate([y_GLM_train,y_GLM_val])
residuals = dataset.std_demand[start_pos:val_pos] - y_GLM


# %%
# PLOT GLM

x_axis = range(start_pos, end_pos)
demand_plt = pd.Series(dataset.demand[start_pos:end_pos],index=x_axis)
demand_pred = destd(y_GLM_train,M,m)
demand_pred_plt = pd.Series(demand_pred,index=x_axis)

plt.figure()
demand_plt.plot()
demand_pred_plt.plot()
plt.show()

# %%

from rmse_f import rmse
print('RMSE_GLM')
print(rmse(dataset.demand[end_pos:val_pos],destd(y_GLM_val,M,m)))


# %% NAX Model
# Needed data stored in a DataFrame

x_axis = range(len(regressors))
names = [str(i) for i in range(9)]
calendar_var = pd.DataFrame(regressors, index = x_axis, columns = names)
calendar_var_NAX = calendar_var[start_pos:val_pos]
temp_data = pd.DataFrame({'std_demand': dataset.std_demand,
                        'log_demand': dataset.log_demand,
                          'residuals': residuals,
                          'drybulb': dataset.drybulb,
                          'dewpnt': dataset.dewpnt})
temp_data_NAX = temp_data[start_pos:val_pos]
df_NAX = pd.concat([temp_data_NAX ,calendar_var_NAX],axis=1)

# %% KONRAD ex. 4
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
#import key packageskeras.utils.plot_model(model, "my_first_model.png")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

from tf_ts_functions import  plot_train_history, multivariate_data
from NAX_functions import custom_loss, inverse_std
from tensorflow.keras.callbacks import EarlyStopping

from NAX_f import prep_data, aggregate_data, NAX_model, demands


mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False


# # %% LOAD DATA
# df_NAX = pd.read_csv("train_data.csv",index_col=0) 
# df_NAX.head() #length 1095

# %% PLOT AND STANDARDIZE
# RMSE_NAX 15776.347510314612 with sigmoid

START_SPLIT = 0
TRAIN_SPLIT = 1095
VAL_SPLIT = 1095+365
BATCH_SIZE = 50 #None
BUFFER_SIZE = 5

EVALUATION_INTERVAL = 500
EPOCHS = 20 #200
REG_PARAM = 0.0001
ACT_FUN = 'softmax' #'sigmoid' 'softmax'
LEARN_RATE = 0.003
HIDDEN_NEURONS=3 #3
LOSS_FUNCTION =  custom_loss #custom_loss #'mae', 'mse'
OUTPUT_NEURONS= 2 #2
STOPPATIENCE = 10

past_history = 2
future_target = -1
STEP = 1

# opt=tf.keras.optimizers.RMSprop()
tf.random.set_seed(14)
features,labels= prep_data(df_NAX,
                    START_SPLIT = START_SPLIT,
                    TRAIN_SPLIT = TRAIN_SPLIT,
                    VAL_SPLIT = VAL_SPLIT)
x_train, y_train,x_val, y_val = aggregate_data(features,labels,
                                  START_SPLIT = START_SPLIT,
                                  TRAIN_SPLIT = TRAIN_SPLIT,
                                  VAL_SPLIT = VAL_SPLIT,
                                  past_history = past_history,
                                  future_target = future_target,
                                  STEP = STEP)


print ('Single window of past history : {}'.format(x_train[0].shape))

# %%

model = NAX_model(INPUT_SHAPE=x_train.shape[-2:],
            REG_PARAM = REG_PARAM,
            ACT_FUN = ACT_FUN,
            LEARN_RATE = LEARN_RATE,
            HIDDEN_NEURONS=HIDDEN_NEURONS ,
            OUTPUT_NEURONS= OUTPUT_NEURONS,
            LOSS_FUNCTION =  LOSS_FUNCTION)


# %%

# %%
EARLYSTOP = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=STOPPATIENCE)
history=model.fit(
    x=x_train, y=y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, callbacks=[EARLYSTOP],
    validation_data=(x_val,y_val), validation_batch_size=BATCH_SIZE,shuffle=True
)

plot_train_history(history,"Loss of model")
# %%
y_pred =model.predict(x_val)
START = TRAIN_SPLIT+past_history+future_target
demand_true, demand_NAX, demand_GLM  = demands(y_pred,y_val, df_NAX,START,M,m)

plt.figure()
demand_true.plot()
demand_NAX.plot()
demand_GLM.plot()
plt.show()

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
print('RMSE_GLM',rmse(demand_GLM, demand_true))
print('RMSE_NAX',rmse(demand_NAX, demand_true))
# %%
# HYPER PARAMETERS READY
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



# %%
