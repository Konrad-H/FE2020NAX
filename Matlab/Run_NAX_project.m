
%% Dataset extraction and datamining
tic
dataset = data_mining("C:/Users/admin/Desktop/Desktop/Politecnico/Quarto anno/Secondo semestre/Financial engineering/Laboratori/Project7NAX/ProjectNAX/GitHub/FE2020NAX/gefcom.csv")
toc

% Dataset description
% DA FARE 
print(dataset[['demand', 'drybulb', 'dewpnt']].describe())

start_date = 2009 
end_date   = 2010
start_pos = (start_date -2008)*365
end_pos   = (end_date+1 -2008)*365

% DA FARE
dataset_plt=dataset[start_pos:end_pos]
plt.figure()
plt.plot(dataset_plt.demand, color = 'red', linewidth=0.5, label='Consumption')
plt.plot(dataset_plt.demand[dataset_plt.day_of_week=='Sun'].index, dataset_plt.demand[dataset_plt.day_of_week=='Sun'].values, 
        linestyle = '', color = 'blue', marker = '.', markersize = 5, label='Sundays')
plt.legend()
gplt.show()
plt.xticks per valori sull'asse x

% Dataset numeric variables are standardized
dataset = data_standardize(dataset);

% Maximum and minimum of log demand, to restore real demand from standardized one 
M = max(dataset[:,2]); % 2:LOG_DEMAND
m = min(dataset[:,2]); % 2:LOG_DEMAND

%% GLM Model
% Define regressors of the GLM
regressors = regressor(dataset);

% GLM on 2008-2010 time window
start_date = 2009;
end_date   = 2011;
val_date   = 2012;
start_pos = (start_date -2008)*365;
end_pos   = (end_date+1 -2008)*365;
val_pos   = (val_date+1 -2008)*365;
[y_GLM_val, y_GLM_train, sigma] = GLM(dataset, regressors, start_date, end_date, val_date); % predicted values on validation set and train set
                                                                                            % and standard deviation of the fitted model
y_GLM = [y_GLM_train,y_GLM_val];
residuals = dataset[start_pos+1:val_pos,3] - y_GLM; % 3: std_demand

% GLM plot
demand_pred = destd(y_GLM_train,M,m);
x_axis = [start_pos+1: end_pos];
figure(1)
plot(x_axis, dataset[start_pos+1:end_pos,1])
hold on
plot(x_axis, demand_pred[start_pos+1:end_pos])

% Plot autocorrelation and partial autocorrelation of the residuals
% DA FARE
residuals_plt = dataset.std_demand[start_pos:end_pos] - y_GLM_train
 
plot_acf(residuals_plt, lags = range(0,51), alpha = None)
plt.xlabel('Days')
#plt.show()
 
plot_pacf(residuals_plt, lags = range(0,51), alpha = None)
plt.xlabel('Days')
#plt.show()

%% NAX Model
% Needed data stored in a DataFrame
x_axis = [1:size(regressors,1)];
calendar_var = regressors[:] % bisogna prendere tutte le colonne tranne quella di uno
calendar_var_NAX = calendar_var[start_pos+1:val_pos];
temp_data = dataset[:] % prendere colonne std_demand, demand, drybulb e dwepnt
temp_data_NAX = temp_data[start_pos+1:val_pos];
df_NAX = [temp_data_NAX ,calendar_var_NAX];

# %% LETI ex. 4 - - DA SISTEMARE DOPO AVERE NAX
# %% IMPORT HYPER PARAM FUNCTION AND LOAD CONSTANTS
from hyper_param_f import find_hyperparam

MAX_EPOCHS = 500;
STOPPATIENCE = 50;

LIST_HIDDEN_NEURONS = [3, 4, 5, 6];
LIST_ACT_FUN = ['softmax', 'sigmoid'];
LIST_LEARN_RATE = [0.1, 0.01, 0.003, 0.001];
LIST_BATCH_SIZE = [50,5000];
LIST_REG_PARAM = [0.001, 0.0001, 0];

START_SPLIT = 0;
TRAIN_SPLIT = 1095;
VAL_SPLIT = 1095+365;

VERBOSE = 1;
VERBOSE_EARLY = 1;

# %%

# BEST COMBINATIOS
# 3.65 %  -- [3, 'softmax', 0.01, 0.001, 50]
# 7752.083425213432
# 4.17 %  -- [3, 'softmax', 0.01, 0.001, 5000]
# Epoch 00218: early stopping
# 7718.890737165838

#min_hyper_parameters, min_RMSE, all_RMSE = find_hyperparam(df_NAX,
#                    MAX_EPOCHS = MAX_EPOCHS, #
#                    STOPPATIENCE = STOPPATIENCE,
#                    LIST_HIDDEN_NEURONS = LIST_HIDDEN_NEURONS, #[3, 4, 5, 6]
#                    LIST_ACT_FUN =LIST_ACT_FUN, #['softmax', 'sigmoid']
#                    LIST_LEARN_RATE = LIST_LEARN_RATE, #[0.1, 0.01, 0.003, 0.001]
#                    LIST_BATCH_SIZE = LIST_BATCH_SIZE, # manca no batch, None non funziona
#                    LIST_REG_PARAM = LIST_REG_PARAM,
#                    VERBOSE= VERBOSE,
#                    VERBOSE_EARLY = VERBOSE_EARLY,
#                    M = M, m = m)
#print(min_hyper_parameters)
#print(min_RMSE)

# %% STORED FOR EASY ACCESS
#all_RMSE = np.load("C:/Users/admin/Desktop/Desktop/Politecnico/Quarto anno/Secondo semestre/Financial engineering/Laboratori/Project7NAX/ProjectNAX/GitHub/FE2020NAX/all_RMSE_1.npy")
#argmin = np.unravel_index(np.argmin(all_RMSE,axis=None),all_RMSE.shape)
#min_hyper_parameters = [ LIST_HIDDEN_NEURONS[argmin[0]],
#                        LIST_ACT_FUN[argmin[1]], 
#                        LIST_LEARN_RATE[argmin[2]], 
#                        LIST_REG_PARAM[argmin[3]],
#                        LIST_BATCH_SIZE[argmin[4]] ]
#min_RMSE = np.min(all_RMSE,axis=None)

# %% Choose Hyperparameters

HIDDEN_NEURONS = 3 #min_hyper_parameters[0] # ??
ACT_FUN = 'softmax' #min_hyper_parameters[1] # ??
LEARN_RATE = 0.003 #min_hyper_parameters[2] # ??
REG_PARAM = 1e-4 #min_hyper_parameters[3] # ??
BATCH_SIZE = 50 #min_hyper_parameters[4] # ??


% Parameters calibration and Confidence interval on the test set. Train set: 2009 - 2011. Test set: 2012 
start_date = 2009;
end_date   = 2011;
test_date  = 2012;
start_pos = (start_date -2008)*365;
end_pos   = (end_date+1 -2008)*365;
test_pos  = (test_date+1 -2008)*365;
[y_GLM_test, y_GLM_train, sigma_GLM] = GLM(dataset, regressors, start_date, end_date, test_date); %predicted values
y_GLM = [y_GLM_train, y_GLM_test];
residuals = dataset.std_demand[start_pos+1:test_pos] - y_GLM;

calendar_var_NAX = calendar_var[start_pos+1:test_pos];
temp_data = pd.DataFrame({'std_demand': dataset.std_demand, % DA RIFARE
                        'log_demand': dataset.log_demand,
                              'residuals': residuals,
                              'drybulb': dataset.drybulb,
                              'dewpnt': dataset.dewpnt})                             
temp_data_NAX = temp_data[start_pos+1:test_pos]
dataset_NAX = [temp_data_NAX ,calendar_var_NAX]

% NAX
MAX_EPOCHS = 600;
STOPPATIENCE = 100;
VERBOSE=1;
VERBOSE_EARLY = 1;

rng(5)
% DA RIFARE
y_pred,history,model = one_NAX_iteration(dataset_NAX,
                    BATCH_SIZE = BATCH_SIZE,
                    EPOCHS = MAX_EPOCHS,
                    REG_PARAM = REG_PARAM,
                    ACT_FUN = ACT_FUN,
                    LEARN_RATE = LEARN_RATE,
                    HIDDEN_NEURONS=HIDDEN_NEURONS ,
                    STOPPATIENCE = STOPPATIENCE,
                    VERBOSE= VERBOSE,
                    VERBOSE_EARLY = VERBOSE_EARLY)
plot_train_history(history,"Loss of model")

%%
mu_NAX = y_pred[:,1];
sigma_NAX = y_pred[:,2];

sigma_NAX = abs(sigma_NAX)
sigma_NAX = clip(sigma_NAX, np.sqrt(0.001), +Inf)

% Confidence interval
y_NAX_test = y_GLM_test[1:] + mu_NAX

[y_NAX_l, y_NAX_u] = ConfidenceInterval(y_NAX_test, sigma_NAX, 0.95, M, m)

x_axis = [end_pos+1:test_pos]
estimated_values = destd(y_NAX_test, M, m);

figure()
hold on
plot(x_axis, dataset[end_pos+1:test_pos,1], 'b', 'LineWidth', 1.2) % prendere colonna demand
plot(x_axis, y_NAX_l, 'r', 'LineWidth', 0.4)
plot(x_axis, y_NAX_u, 'r', 'LineWidth', 0.4)
plot(x_axis, estimated_values, 'r', 'LineWidth', 0.8)
x_area = [x, fliplr(x)];
y_area = [y_NAX_l, fliplr(y_NAX_u)];
fill(x_area, y_area, 'y');


% ex. 6

for i =[0:4]
    start_date = 2010+i
    end_date   = 2012+i
    test_date  = 2013+i

    %start_date = 2009
    %end_date   = 2011
    %test_date  = 2012
    start_pos = (start_date -2008)*365;
    end_pos   = (end_date+1 -2008)*365;
    val_pos   = (val_date+1 -2008)*365;
    [y_GLM_val, y_GLM_train, sigma] = GLM(dataset, regressors, start_date, end_date, val_date); % predicted values on validation set and train set
                                                                                                % and standard deviation of the fitted model
    y_GLM = [y_GLM_train,y_GLM_val];
    residuals = dataset[start_pos+1:val_pos,3] - y_GLM; % 3: std_demand

    disp('RMSE_GLM')
    rmse(dataset.demand[end_pos+1:test_pos],destd(y_GLM_test, M, m)))
    disp('MAPE_GLM')
    mape(dataset.demand[end_pos+1:test_pos],destd(y_GLM_test, M, m)))

    % DA QUI IN POI ANCORA DA SISTEMARE

    calendar_var_NAX = calendar_var[start_pos:test_pos]
    temp_data = pd.DataFrame({'std_demand': dataset.std_demand,
                                'residuals': residuals,
                                'drybulb': dataset.drybulb,
                                'dewpnt': dataset.dewpnt})                             
    temp_data_NAX = temp_data[start_pos:test_pos]
    dataset_NAX = pd.concat([temp_data_NAX ,calendar_var_NAX],axis=1)

    MAX_EPOCHS = 500 
    STOPPATIENCE = 50
    VERBOSE=0
    VERBOSE_EARLY=1
    y_pred,history,model = one_NAX_iteration(dataset_NAX,
                        BATCH_SIZE = BATCH_SIZE,
                        EPOCHS = MAX_EPOCHS,
                        REG_PARAM = REG_PARAM,
                        ACT_FUN = ACT_FUN,
                        LEARN_RATE = LEARN_RATE,
                        HIDDEN_NEURONS=HIDDEN_NEURONS ,
                        STOPPATIENCE = STOPPATIENCE,
                        VERBOSE= VERBOSE,
                        VERBOSE_EARLY = VERBOSE_EARLY)
    plot_train_history(history,"Loss of model")

    mu_NAX = y_pred[:,0]
    sigma_NAX = y_pred[:,1]
    sigma_NAX = abs(sigma_NAX)
    sigma_NAX = np.clip(sigma_NAX, np.sqrt(0.001), None)

    y_NAX_test = y_GLM_test[1:] + mu_NAX

    from ConfidenceInterval_f import ConfidenceInterval
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

    print('RMSE_NAX')
    print(rmse(dataset.demand[end_pos+1:test_pos],destd(y_NAX_test, M, m)))
    print('MAPE_NAX')
    print(mape(dataset.demand[end_pos+1:test_pos],destd(y_NAX_test, M, m)))

    # %% ARX Model
    from ARX_f import ARX

    y_ARX_test, sigma_ARX = ARX(dataset_NAX, start_date, end_date, test_date)

    print('RMSE_ARX')
    print(rmse(dataset.demand[end_pos:test_pos],destd(y_ARX_test, M, m)))
    print('MAPE_ARX')
    print(mape(dataset.demand[end_pos:test_pos],destd(y_ARX_test, M, m)))

    # %% pinball
    from pinball_f import pinball
    from backtest_f import backtest

    y = np.array(dataset.demand[end_pos:test_pos])  # demand, non std_demand!!!!!
    y_ARX_test = np.array(y_ARX_test)

    pinball_values_GLM = pinball(y, y_GLM_test, sigma_GLM, M, m)   # y_pred = output of GLM (prediction of std(log_demand))
    pinball_values_NAX = pinball(y[1:], y_NAX_test, sigma_NAX, M, m)
    pinball_values_ARX = pinball(y, y_ARX_test, sigma_ARX, M, m)

    pinplot_GLM = pd.Series(pinball_values_GLM)
    pinplot_NAX = pd.Series(pinball_values_NAX)
    pinplot_ARX = pd.Series(pinball_values_ARX)

    # pinball graph
    plt.figure()
    pinplot_GLM.plot()
    pinplot_NAX.plot()
    pinplot_ARX.plot()
    plt.show()

    # %% backtest
    from backtest_f import backtest


    print('backtest')
    confidence_levels = np.arange(0.9,1,0.01)
    backtested_levels_GLM, LR_Unc_GLM, LR_Cov_GLM = backtest(y, y_GLM_test, confidence_levels, sigma_GLM, M, m)
    backtested_levels_NAX, LR_Unc_NAX, LR_Cov_NAX = backtest(y[1:], y_NAX_test, confidence_levels, sigma_NAX, M, m)
    backtested_levels_ARX, LR_Unc_ARX, LR_Cov_ARX = backtest(y, y_ARX_test, confidence_levels, sigma_ARX, M, m)

    print('LR_GLM')
    print(LR_Unc_GLM, LR_Cov_GLM)
    print('LR_NAX')
    print(LR_Unc_NAX, LR_Cov_NAX)
    print('LR_ARX')
    print(LR_Unc_ARX, LR_Cov_ARX)

    backplot_GLM = pd.Series(backtested_levels_GLM)
    backplot_NAX = pd.Series(backtested_levels_NAX)
    backplot_ARX = pd.Series(backtested_levels_ARX)
    confplot     = pd.Series(confidence_levels)

    plt.figure()
    backplot_GLM.plot()
    backplot_NAX.plot()
    backplot_ARX.plot()
    confplot.plot()
    plt.show()
end

