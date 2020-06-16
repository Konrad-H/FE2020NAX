# To add a new cell, type '# %%'

# %% 
# Import useful built-in packages
import pandas as pd 
import numpy as np
import os
import sys
from tensorflow.random import set_seed
import time
import matplotlib.pyplot as plt

from standard_and_error_functions import destd
# %% 
# Dataset extraction and datamining
from data_mining_f import data_mining, data_standardize

# tic = time.time()
# dataset = data_mining("../gefcom.csv")
# toc = time.time()
# dataset.to_csv('dataset.csv')

dataset = pd.read_csv('dataset.csv')
# print(str(toc-tic) + ' sec Elapsed\n')

# Summary of energy demand and weather variables
print(dataset.head())
print()
print(dataset[['demand', 'drybulb', 'dewpnt']].describe())
 
# %%
# PLOT
# Graph representing demand over 2009 and 2010
# Sundays are highlighted by blue marker
plot =False
if plot:
        start_date = 2009 
        end_date   = 2010
        start_pos = (start_date -2008)*365
        end_pos   = (end_date+1 -2008)*365
        dataset_plt = dataset[start_pos:end_pos+1]

        plt.figure()
        plt.plot(dataset_plt.demand.index, dataset_plt.demand/1000, color='red', linewidth=0.5, label='Consumption')
        plt.plot(dataset_plt.demand[dataset_plt.day_of_week=='Sun'].index, dataset_plt.demand[dataset_plt.day_of_week=='Sun'].values/1000, 
                linestyle='', color='blue', marker='.', markersize=5, label='Sundays')
        plt.legend()
        plt.xticks(np.array([dataset_plt[dataset_plt.date=='2009-01-01'].index, dataset_plt[dataset_plt.date=='2009-04-01'].index, 
                dataset_plt[dataset_plt.date=='2009-07-01'].index, dataset_plt[dataset_plt.date=='2009-10-01'].index,
                dataset_plt[dataset_plt.date=='2010-01-01'].index, dataset_plt[dataset_plt.date=='2010-04-01'].index,
                dataset_plt[dataset_plt.date=='2010-07-01'].index, dataset_plt[dataset_plt.date=='2010-10-01'].index,
                dataset_plt[dataset_plt.date=='2011-01-01'].index]),
                ['2009-01', '2009-04', '2009-07', '2009-10', '2010-01', '2010-04', '2010-07', '2010-10', '2011-01'],
                fontsize='small')
        plt.ylabel('GWh')
        plt.show()

# %% 
# Numeric variables are standardized, mapping them in [0,1] 
dataset = data_standardize(dataset)

# Maximum and minimum values taken by log_demand are saved, as they are useful to go 
# back from standardized values to demand values, using custom function destd

M = max(dataset.log_demand)
m = min(dataset.log_demand)

# %% 
# GLM Model
from GLM_and_ARX_models import regressor, GLM

# define regressors
regressors = regressor(dataset)

# GLM on 2008-2010 time window
start_date = 2008
end_date   = 2010
val_date   = 2011
start_pos = (start_date -2008)*365
end_pos   = (end_date+1 -2008)*365
val_pos   = (val_date+1 -2008)*365
y_GLM_val, y_GLM_train, sigma = GLM(dataset, regressors, start_date, end_date, val_date) # predicted values
y_GLM = np.concatenate([y_GLM_train,y_GLM_val])
residuals = dataset.std_demand[start_pos:val_pos] - y_GLM # model residuals

# %%
# GLM plot on train test - uncomment to plot it
if plot:
        x_axis = range(start_pos, end_pos)
        demand_pred = destd(y_GLM_train,M,m)

        demand_plt = pd.Series(dataset.demand[start_pos:end_pos],index=x_axis)
        demand_pred_plt = pd.Series(demand_pred,index=x_axis)

        plt.figure()
        demand_plt.plot()
        demand_pred_plt.plot()
        plt.show()

# %%
# # Plot autocorrelation and partial autocorrelation of the residuals
if plot:
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

        residuals_plt = dataset.std_demand[start_pos:end_pos] - y_GLM_train

        # Autocorrelation
        plot_acf(residuals_plt, lags = range(0,51), alpha = None)
        plt.xlabel('Days')
        plt.show()

        # Partial Autocorrelation
        plot_pacf(residuals_plt, lags = range(0,51), alpha = None)
        plt.xlabel('Days')
        plt.show()

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


# %% Selection of the optimal hyper-parameters (corresponding to the minimum RMSE)
from hyper_param_f import find_hyperparam
from MLE_loss import loss_strike
from tensorflow.keras import initializers
MAX_EPOCHS = 500
STOPPATIENCE = 50

strike = 0.0001
my_loss,y2var = loss_strike(strike)


START_SPLIT = 0
TRAIN_SPLIT = 1095
VAL_SPLIT = 1095+365

VERBOSE = 1
VERBOSE_EARLY = 1

# Possible values of hyper-parameters
#hyper_grid =int(input('What grid style?  - 1 standard; 2 simplifie; 3 extended; 4 extendend 2.0')) 
hyper_grid = 1
if hyper_grid==1:
        LIST_HIDDEN_NEURONS = [[3], [4], [5],[6]]  
        LIST_ACT_FUN = ['softmax', 'sigmoid']   # activation function
        LIST_LEARN_RATE = [  0.001, 0.003,0.01,0.1]     # initial learning rate (for Keras ADAM)
        LIST_REG_PARAM = [0.001, 0.0001, 0]     # regularization parameter
        LIST_BATCH_SIZE = [50, 5000]     # batch size, 5000 for no batch

elif  hyper_grid==2:
        LIST_HIDDEN_NEURONS = [[3]]  
        LIST_ACT_FUN = ['softmax', 'sigmoid']   # activation function
        LIST_LEARN_RATE = [0.001]     # initial learning rate (for Keras ADAM)
        LIST_REG_PARAM = [0.001]     # regularization parameter
        LIST_BATCH_SIZE = [50, 5000]     # batch size, 5000 for no batch

elif hyper_grid==3:
        BASE_HIDDEN_NEURONS = [3,4,5,6,8,10]
        BASE_HIDDEN_NEURONS = [[i] for i in BASE_HIDDEN_NEURONS]
        MULTI_LAYER_HIDDEN = [[3,3],[3,3,3],[3,10],[10,3]]
        LIST_HIDDEN_NEURONS = BASE_HIDDEN_NEURONS+MULTI_LAYER_HIDDEN   # number of neurons (hidden layer)
        LIST_ACT_FUN = ['softmax','sigmoid',  'tanh', 'softplus']   # activation function
        LIST_LEARN_RATE = [0.0001, 0.001, 0.01, 0.1]     # initial learning rate (for Keras ADAM)
        LIST_REG_PARAM = [0, 0.0001  , 0.01, 0.001]    # regularization parameter
        LIST_BATCH_SIZE = [25, 50, 500, 5000]     # batch size, 5000 for no batch

elif hyper_grid==4:
        BASE_HIDDEN_NEURONS = [3,4,5,6,8,10]
        BASE_HIDDEN_NEURONS = [[i] for i in BASE_HIDDEN_NEURONS]
        MULTI_LAYER_HIDDEN = [[3,3],[3,3,3], [3,4],[5,4,3]]
        LIST_HIDDEN_NEURONS = BASE_HIDDEN_NEURONS+MULTI_LAYER_HIDDEN   # number of neurons (hidden layer)
        LIST_ACT_FUN = ['softmax','sigmoid',  'tanh', 'softplus']   # activation function
        LIST_LEARN_RATE = [0.0001, 0.001, 0.003, 0.01, 0.03, 0.1]     # initial learning rate (for Keras ADAM)
        LIST_REG_PARAM = [0, 0.0001, 0.0003, 0.01, 0.001]    # regularization parameter
        LIST_BATCH_SIZE = [25, 50, 100, 500, 5000]     # batch size, 5000 for no batch

piece_run = False
K = 4
N = len(LIST_HIDDEN_NEURONS)

if piece_run:
        i= int(input('This run is very big, choose a partition between 1 and '+str(K)+':'))
        LIST_HIDDEN_NEURONS = LIST_HIDDEN_NEURONS[((i-1)*N//K):(i*N//K)]



# Hyperparam run
seed = 0 #301
#seed = int(input('What seed?'))
set_seed(seed)
name = 'Results/RMSE.'+str(seed)+'.'+str(strike)

live_run = False
save = True
if live_run:
        hid_ker_init = 'zeros' #'glorot_uniform' 
        out_ker_init= 'zeros'
        # hid_bias_init = initializers.RandomUniform(minval=-2/1000, maxval=2/1000)
        hid_bias_init= 'zeros'
        out_bias_init= initializers.Constant([0,0.1])
        #out_bias_init = 'zeros' #'zeros'
        # out_bias_init = initializers.RandomUniform(minval=-2/1000, maxval=2/1000)
        if hyper_grid==1 or hyper_grid==2:
                
                all_RMSE, model = find_hyperparam(df_NAX, M = M, m = m,
                                                LOSS_FUNCTION = my_loss,
                                                Y2VAR = y2var,
                                                MAX_EPOCHS = MAX_EPOCHS,
                                                STOPPATIENCE = STOPPATIENCE,
                                                LIST_HIDDEN_NEURONS = LIST_HIDDEN_NEURONS,
                                                LIST_ACT_FUN = LIST_ACT_FUN,
                                                LIST_LEARN_RATE = LIST_LEARN_RATE,
                                                LIST_BATCH_SIZE = LIST_BATCH_SIZE,
                                                LIST_REG_PARAM = LIST_REG_PARAM,
                                                VERBOSE = VERBOSE,
                                                VERBOSE_EARLY = VERBOSE_EARLY,
                                                OUT_KERNEL = out_ker_init,
                                                OUT_BIAS = out_bias_init,
                                                HID_KERNEL = hid_ker_init,
                                                HID_BIAS = hid_ker_init)
                hid_weights = model.layers[0].get_weights()
                out_weights = model.layers[1].get_weights()
                if save:
                        array = np.array([all_RMSE,hid_weights,out_weights ])
                        np.save(name+'.npy', array)
        elif hyper_grid==3 or hyper_grid==4:
                for layer_n in LIST_HIDDEN_NEURONS:
                        all_RMSE, model = find_hyperparam(df_NAX, M = M, m = m,
                                                        LOSS_FUNCTION = my_loss,
                                                        Y2VAR = y2var,
                                                        MAX_EPOCHS = MAX_EPOCHS,
                                                        STOPPATIENCE = STOPPATIENCE,
                                                        LIST_HIDDEN_NEURONS = [layer_n],
                                                        LIST_ACT_FUN = LIST_ACT_FUN,
                                                        LIST_LEARN_RATE = LIST_LEARN_RATE,
                                                        LIST_BATCH_SIZE = LIST_BATCH_SIZE,
                                                        LIST_REG_PARAM = LIST_REG_PARAM,
                                                        VERBOSE = VERBOSE,
                                                        VERBOSE_EARLY = VERBOSE_EARLY,
                                                        OUT_KERNEL = out_ker_init,
                                                        OUT_BIAS = out_bias_init,
                                                        HID_KERNEL = hid_ker_init,
                                                        HID_BIAS = hid_ker_init)
                        weights = []
                        hid_weights = model.layers[0].get_weights()
                        out_weights = model.layers[-1].get_weights()                
                        for i in range(len(layer_n)+1):
                                weights.append(model.layers[i].get_weights())

                        if save:
                                
                                name = 'Results/RMSE_grid'+str(hyper_grid)+'neur'+str(layer_n)+'seed'+str(seed)
                                array = np.array([all_RMSE,weights ])
                                np.save(name+'.npy', array)

else:
        if hyper_grid==1 or hyper_grid==2:
                data = np.load(name+'.npy', allow_pickle=True)
                all_RMSE = data[0]
                hid_weights = data[1]
                out_weights = data[2]
        if hyper_grid==3:
                all_data = []
                for layer_n in LIST_HIDDEN_NEURONS:
                        name = 'Results/RMSE'+str(hyper_grid)+'neur'+str(layer_n)+''+str(seed)
                        data = np.load(name+'.npy', allow_pickle=True)
                        all_data.append(data)
                all_RMSE = []
                all_weights=[]
                for i in range(len(all_data)):
                        all_RMSE+=[all_data[i][0][0]]
                        all_weights+=[all_data[i][1]]
                all_RMSE = np.array(all_RMSE)
                hid_weights = all_weights[2][0]
                out_weights = all_weights[2][1]



# %% 
# summary
plt.hist(all_RMSE.flatten()*(all_RMSE.flatten()<30000) + 30001*(all_RMSE.flatten()>30000))
plt.xlabel('RMSE')
plt.savefig('Plots/HisogramEs8.Seed'+str(seed)+'.png')
argmin = np.unravel_index(np.argmin(all_RMSE,axis=None),all_RMSE.shape)
min_hyper_parameters = [LIST_HIDDEN_NEURONS[argmin[0]],
                        LIST_ACT_FUN[argmin[1]], 
                        LIST_LEARN_RATE[argmin[2]], 
                        LIST_REG_PARAM[argmin[3]],
                        LIST_BATCH_SIZE[argmin[4]]]
min_RMSE = np.min(all_RMSE,axis=None)
print(min_RMSE,' -- ' ,min_hyper_parameters)

# %%
if True:
        k = 10
        idx = np.argpartition(all_RMSE.flatten(), k)
        best_values = np.zeros((k,5)).tolist()
        for i in range(k):
                
                ith_value = np.unravel_index(idx[i],all_RMSE.shape)
                temp_hyper_parameters = [LIST_HIDDEN_NEURONS[ith_value[0]],
                                LIST_ACT_FUN[ith_value[1]], 
                                LIST_LEARN_RATE[ith_value[2]], 
                                LIST_REG_PARAM[ith_value[3]],
                                LIST_BATCH_SIZE[ith_value[4]]]
                best_values[i][0]=all_RMSE.flatten()[idx[i]]
                best_values[i][1:6]=temp_hyper_parameters
        best_values = sorted(best_values,key=lambda x: (x[0]))
        col_names = ['RMSE','Hid Neurons','Act Fun', 'Learn Rate', 'Reg Param', 'Batch Size']
        df_best = pd.DataFrame(best_values, columns=col_names)
        TH = 10000
        df_best = df_best[df_best['RMSE']<TH]
        for i in range(6):
                plt.figure()
                # plt.subplot(2, 3, 1+i)
                if i==0:
                        plt.hist(df_best[col_names[i]]/1000)
                else:
                        df_best[col_names[i]].value_counts().plot(kind='bar')
                plt.savefig('Plots/Es8barplots'+str(i)+'_K.png')
                plt.show()
        # plt.show()
# %% Choose Hyperparameters

HIDDEN_NEURONS = min_hyper_parameters[0] # ??
ACT_FUN = min_hyper_parameters[1] # ??
LEARN_RATE = min_hyper_parameters[2] # ??
REG_PARAM = min_hyper_parameters[3] # ??
BATCH_SIZE = min_hyper_parameters[4] # ??


hid_kernel_hyp = hid_weights[0]
hid_rec_hyp = hid_weights[1]
hid_bias_hyp  = hid_weights[2]
out_kernel_hyp  = out_weights[0]
out_bias_hyp  = out_weights[1]


# HYPER PARAMETERS READY
# %% ex. 5
#
start_date = 2008
end_date   = start_date+2
test_date  = start_date+3
start_pos = (start_date -2008)*365
end_pos   = (end_date+1 -2008)*365
test_pos  = (test_date+1 -2008)*365
y_GLM_test, y_GLM_train, sigma_GLM = GLM(dataset, regressors, start_date, end_date, test_date) #predicted values
y_GLM = np.concatenate([y_GLM_train, y_GLM_test])
residuals = dataset.std_demand[start_pos:test_pos] - y_GLM

calendar_var_NAX = calendar_var[start_pos:test_pos]
temp_data = pd.DataFrame({'std_demand': dataset.std_demand,
                        'log_demand': dataset.log_demand,
                              'residuals': residuals,
                              'drybulb': dataset.drybulb,
                              'dewpnt': dataset.dewpnt})                             
temp_data_NAX = temp_data[start_pos:test_pos]
dataset_NAX = pd.concat([temp_data_NAX ,calendar_var_NAX],axis=1)


# %%
from NAX_f import one_NAX_iteration, plot_train_history
from tensorflow.keras.initializers import Constant
# Loss function used after hyperparam found

set_seed(501)
set_seed(10)

MLE_loss, y2var = loss_strike(strike)

MAX_EPOCHS = 600 
STOPPATIENCE = 50
VERBOSE = 1
VERBOSE_EARLY = 1

y_pred,history,model = one_NAX_iteration(dataset_NAX,
                        BATCH_SIZE = BATCH_SIZE,
                        EPOCHS = MAX_EPOCHS,
                        REG_PARAM = REG_PARAM,
                        ACT_FUN = ACT_FUN,
                        LEARN_RATE = LEARN_RATE,
                        HIDDEN_NEURONS=HIDDEN_NEURONS ,
                        STOPPATIENCE = STOPPATIENCE,
                        VERBOSE= VERBOSE,
                        VERBOSE_EARLY = VERBOSE_EARLY,
                        LOSS_FUNCTION = MLE_loss,
                        OUT_KERNEL = Constant(out_kernel_hyp ),
                        OUT_BIAS = Constant(out_bias_hyp ),
                        HID_KERNEL = Constant(hid_kernel_hyp ),
                        HID_BIAS = Constant(hid_bias_hyp ),
                        HID_REC = Constant(hid_rec_hyp)
                    )
plot_train_history(history,"Loss of model")

mu_NAX = y_pred[:,0]
sigma_NAX = np.sqrt(y2var(y_pred))
sigma_NAX = sigma_NAX[:,0]
print('MAX sigma: ', max(sigma_NAX))
print('MIN sigma: ', min(sigma_NAX))

# %% Confidence interval
from evaluation_functions import ConfidenceInterval
from standard_and_error_functions import rmse, mape

y_NAX_test = y_GLM_test[1:] + mu_NAX
y_test = dataset.demand[end_pos+1:test_pos]
print('RMSE_NAX')
print(rmse(y_test,destd(y_NAX_test, M, m)))
    
# Confidence Interval at confidence level 95% 

y_NAX_l, y_NAX_u = ConfidenceInterval(y_NAX_test, sigma_NAX, 0.95, M, m)

print( sum((y_test>y_NAX_l)&(y_test<y_NAX_u))/len(y_NAX_l))


# Plot 95% Confidence Interval
x_axis = range(end_pos+1, test_pos)
lower_bound = pd.Series(y_NAX_l, index=x_axis)
upper_bound = pd.Series(y_NAX_u, index=x_axis)
estimated_values = pd.Series(destd(y_NAX_test, M, m), index=x_axis)
real_values = destd(dataset.std_demand[end_pos+1:test_pos], M, m)
real_values = pd.Series(real_values, index=x_axis)

plt.figure()
plt.plot(x_axis, real_values, '-', color='b', linewidth=1.2)
plt.plot(x_axis, lower_bound, color='r', linewidth=0.4)
plt.plot(x_axis, upper_bound, color='r', linewidth=0.4)
plt.plot(x_axis, estimated_values, color='r', linewidth=0.8)
plt.fill_between(x_axis, lower_bound, upper_bound, facecolor='coral', interpolate=True)
plt.show()

# %%
# Comparison of GLM, NAX and ARX models, through pinball and backtest techniques, and errors MAPE and RMSE
# We calibrate four times the model over a 3 years time window, and test it on the fourth year
# Test years: 2012 - 2013 - 2014 - 2015 - 2016

from GLM_and_ARX_models import ARX

from evaluation_functions import pinball, backtest
set_seed(501)


hid_kernel = hid_kernel_hyp
hid_bias = hid_bias_hyp
hid_rec = hid_rec_hyp
out_kernel = out_kernel_hyp
out_bias = out_bias_hyp

table_col = ['Measure','Anno', 'GLM', 'ARX', 'NAX' ]
results = []
for i in range(5):

    # train and test sets are defined
    start_date = 2009+i
    end_date   = 2011+i
    test_date  = 2012+i
    start_pos = (start_date -2008)*365
    end_pos   = (end_date+1 -2008)*365
    test_pos  = (test_date+1 -2008)*365
    

    # GLM calibration
    y_GLM_test, y_GLM_train, sigma_GLM = GLM(dataset, regressors, start_date, end_date, test_date) #predicted values
    y_GLM = np.concatenate([y_GLM_train, y_GLM_test])
    
    residuals = dataset.std_demand[start_pos:test_pos] - y_GLM

    # Dataset for NAX model is prepared
    calendar_var_NAX = calendar_var[start_pos:test_pos]
    temp_data = pd.DataFrame({'std_demand': dataset.std_demand,
                                'residuals': residuals,
                                'drybulb': dataset.drybulb,
                                'dewpnt': dataset.dewpnt})                             
    temp_data_NAX = temp_data[start_pos:test_pos]
    dataset_NAX = pd.concat([temp_data_NAX ,calendar_var_NAX],axis=1)


    # NAX model is calibrated
    MAX_EPOCHS = 500 
    STOPPATIENCE = 50
    VERBOSE = 0
    VERBOSE_EARLY = 1
    y_pred,history,model = one_NAX_iteration(dataset_NAX,
                        BATCH_SIZE = BATCH_SIZE,
                        EPOCHS = MAX_EPOCHS,
                        REG_PARAM = REG_PARAM,
                        ACT_FUN = ACT_FUN,
                        LEARN_RATE = LEARN_RATE,
                        HIDDEN_NEURONS=HIDDEN_NEURONS ,
                        STOPPATIENCE = STOPPATIENCE,
                        VERBOSE= VERBOSE,
                        VERBOSE_EARLY = VERBOSE_EARLY,
                        LOSS_FUNCTION = MLE_loss,
                        OUT_KERNEL = Constant(out_kernel ),
                        OUT_BIAS = Constant(out_bias ),
                        HID_KERNEL = Constant(hid_kernel ),
                        HID_BIAS = Constant(hid_bias ),
                        HID_REC = Constant(hid_rec)
                        )

    hid_weights = model.layers[0].get_weights()
    hid_kernel = hid_weights[0]
    hid_bias = hid_weights[-1]
    hid_rec = hid_weights[1]

    out_weights = model.layers[1].get_weights()
    out_kernel = out_weights[0]
    out_bias = out_weights[1]

    plot_train_history(history,"Loss of model")
    
    mu_NAX = y_pred[:,0]
    sigma_NAX = np.sqrt(y2var(y_pred))
    sigma_NAX = sigma_NAX[:,0]
    y_NAX_test = y_GLM_test[1:] + mu_NAX

    # 95% Confidence Interval Plot
    y_NAX_l, y_NAX_u = ConfidenceInterval(y_NAX_test, sigma_NAX, 0.95, M, m)

    x_axis = range(end_pos+1, test_pos)
    lower_bound = pd.Series(y_NAX_l, index=x_axis)
    upper_bound = pd.Series(y_NAX_u, index=x_axis)
    estimated_values = pd.Series(destd(y_NAX_test, M, m), index=x_axis)
    real_values = destd(dataset.std_demand[end_pos+1:test_pos], M, m)
    real_values = pd.Series(real_values, index=x_axis)

    plt.figure()
    plt.plot(x_axis, real_values, '-', color='b', linewidth=1.2)
    plt.plot(x_axis, lower_bound, color='r', linewidth=0.4)
    plt.plot(x_axis, upper_bound, color='r', linewidth=0.4)
    plt.plot(x_axis, estimated_values, color='r', linewidth=0.8)
    plt.fill_between(x_axis, lower_bound, upper_bound, facecolor='coral', interpolate=True)
    plt.show()


    # ARX Model is calibrated
    y_ARX_test, sigma_ARX = ARX(dataset_NAX, start_date, end_date, test_date)
    

    # Pinball Loss computed for the three models
    y = np.array(dataset.demand[end_pos:test_pos])
    y_ARX_test = np.array(y_ARX_test)
    
    pinball_values_GLM = pinball(y, y_GLM_test, sigma_GLM, M, m)
    pinball_values_NAX = pinball(y[1:], y_NAX_test, sigma_NAX, M, m)
    pinball_values_ARX = pinball(y, y_ARX_test, sigma_ARX, M, m)

    pinplot_GLM = pd.Series(pinball_values_GLM)
    pinplot_NAX = pd.Series(pinball_values_NAX)
    pinplot_ARX = pd.Series(pinball_values_ARX)

    # Pinball Loss Graph
    plt.figure()
    plt.plot(pinplot_GLM.index/100, pinplot_GLM.values/1000, linestyle='dashed', color='red', label='GLM')
    plt.plot(pinplot_NAX.index/100, pinplot_NAX.values/1000, color='black', label='NAX')
    plt.plot(pinplot_ARX.index/100, pinplot_ARX.values/1000, linestyle='dotted', color='blue', label='ARX')
    plt.legend()
    plt.xlabel('Quantile')
    plt.ylabel('Pinball Loss [GWh]')
    plt.show()

    
    # Backtest
    print('backtest')
    confidence_levels = np.arange(0.9,1,0.01)
    backtested_levels_GLM, LR_Unc_GLM, LR_Cov_GLM = backtest(y, y_GLM_test, confidence_levels, sigma_GLM, M, m)
    backtested_levels_NAX, LR_Unc_NAX, LR_Cov_NAX = backtest(y[1:], y_NAX_test, confidence_levels, sigma_NAX, M, m)
    backtested_levels_ARX, LR_Unc_ARX, LR_Cov_ARX = backtest(y, y_ARX_test, confidence_levels, sigma_ARX, M, m)

    # Likelihood Ratios of Conditional and Unconditional Covarage Test 


    backplot_GLM = pd.Series(backtested_levels_GLM)
    backplot_NAX = pd.Series(backtested_levels_NAX)
    backplot_ARX = pd.Series(backtested_levels_ARX)
    confplot     = pd.Series(confidence_levels)

    # Backtested Levels Graph
    plt.figure()
    plt.plot(confplot.values, backplot_GLM.values, linestyle='dashed', color='red', label='GLM')
    plt.plot(confplot.values, backplot_NAX.values, color='black', label='NAX')
    plt.plot(confplot.values, backplot_ARX.values, linestyle='dotted', color='blue', label='ARX')
    plt.plot(confplot.values, confplot.values, linestyle='', color='cyan', marker='.', markersize=5, label='Nominal Level')
    plt.legend()
    plt.xlabel('Nominal Level')
    plt.ylabel('Backtested Level')
    plt.show()


    print('MAX sigma: ', max(sigma_NAX))
    print('MIN sigma: ', min(sigma_NAX))
    # GLM errors
    print('RMSE_GLM')
    RMSE_GLM=rmse(dataset.demand[end_pos:test_pos],destd(y_GLM_test, M, m))
    print(RMSE_GLM)
    print('MAPE_GLM')
    MAPE_GLM=mape(dataset.demand[end_pos:test_pos],destd(y_GLM_test, M, m))
    print(MAPE_GLM)
    # NAX Errors
    print('RMSE_NAX')
    RMSE_NAX=rmse(dataset.demand[end_pos+1:test_pos],destd(y_NAX_test, M, m))
    print(RMSE_NAX)
    print('MAPE_NAX')
    MAPE_NAX=mape(dataset.demand[end_pos+1:test_pos],destd(y_NAX_test, M, m))
    print(MAPE_NAX)
    # ARX Errors
    print('RMSE_ARX')
    RMSE_ARX=rmse(dataset.demand[end_pos:test_pos],destd(y_ARX_test, M, m))
    print(RMSE_ARX)
    print('MAPE_ARX')
    MAPE_ARX=mape(dataset.demand[end_pos:test_pos],destd(y_ARX_test, M, m))
    print(MAPE_ARX)
    
    APL_GLM = sum(pinball_values_GLM)/len(pinball_values_GLM)
    APL_NAX = sum(pinball_values_NAX)/len(pinball_values_NAX)
    APL_ARX = sum(pinball_values_ARX)/len(pinball_values_ARX)


    RMSE_year = ['RMSE',test_date,RMSE_GLM, RMSE_ARX, RMSE_NAX] 
    MAPE_year = ['MAPE',test_date, MAPE_GLM,MAPE_ARX, MAPE_NAX]
    APL_year = ['APL',test_date, APL_GLM,APL_ARX, APL_NAX]
    LRU_year = ['LRU',test_date, LR_Unc_GLM,LR_Unc_ARX, LR_Unc_NAX]
    LRC_year = ['LRC',test_date, LR_Cov_GLM,LR_Cov_ARX, LR_Cov_NAX]
    print('LR_GLM')
    print(LR_Unc_GLM, LR_Cov_GLM)
    print('LR_NAX')
    print(LR_Unc_NAX, LR_Cov_NAX)
    print('LR_ARX')
    print(LR_Unc_ARX, LR_Cov_ARX)
    results.append(RMSE_year)
    results.append(MAPE_year)
    results.append(APL_year)
    results.append(LRU_year)
    results.append(LRC_year)
ES8_esults = pd.DataFrame(results, columns=table_col)




# %%
