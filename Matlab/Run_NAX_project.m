clear all
clc
close all
rng(14);
%% Dataset extraction and datamining
tic
dataset = data_mining("../gefcom.csv");
toc

% Dataset description
summary(dataset);
mean_demand = mean(dataset.demand);
mean_dewpnt = mean(dataset.dewpnt);
mean_drybulb = mean(dataset.drybulb);

% Graph representing demand over 2009 and 2010
% Sundays are highlighted by blue marker
start_day = datenum('2009-01-01', 'yyyy-mm-dd');
end_day = datenum('2011-01-01', 'yyyy-mm-dd');
start_pos = find(dataset.dates == start_day);
end_pos = find(dataset.dates == end_day);

dataset_plt = dataset(start_pos:end_pos, :);
sundays_pos = find(strcmp(dataset_plt.day_of_week, "Sun"));
dates_to_plot = datenum(['2009-01-01'; '2009-04-01'; '2009-07-01'; '2009-10-01'; '2010-01-01'; '2010-04-01'; '2010-07-01'; '2010-10-01'; '2011-01-01']);
for i = 1:length(dates_to_plot)
    plot_pos(i) = find(dataset_plt.dates == dates_to_plot(i));
end

figure()
plot([start_pos:end_pos], dataset_plt.demand/1000, 'r', ...
      start_pos-1+sundays_pos, dataset_plt.demand(sundays_pos)/1000, 'b.', 'MarkerSize', 8);
xlim([start_pos-30, end_pos+30])
legend('Consumption', 'Sundays', 'Location', 'Northwest')
ylabel('GWh')
xticks(start_pos-1+plot_pos)
xticklabels({'2009-01', '2009-04', '2009-07', '2009-10', '2010-01', '2010-04', '2010-07', '2010-10', '2011-01'})
ax = gca;
ax.FontSize = 8; 

% Dataset numeric variables are standardized, mapping them in [0,1]
dataset = data_standardize(dataset);

% Maximum and minimum of log demand, to restore real demand from standardized one 
M = max(dataset.log_demand);
m = min(dataset.log_demand);


%% GLM Model
% Define regressors of the GLM
regressors = regressor(dataset);

% GLM on 2008-2010 time window
start_date = 2009;
end_date   = 2011;
val_date   = 2012;
start_pos = (start_date - 2008)*365;
end_pos   = (end_date+1 - 2008)*365;
val_pos   = (val_date+1 - 2008)*365;
[y_GLM_val, y_GLM_train, sigma] = GLM(dataset, regressors, start_date, end_date, val_date); % predicted values on validation set and train set
                                                                                            % and standard deviation of the fitted model
y_GLM = [y_GLM_train; y_GLM_val];
residuals = dataset.std_demand(start_pos+1:val_pos) - y_GLM;

% GLM plot
demand_pred = destd(y_GLM_train, M, m);

figure()
plot([start_pos+1: end_pos], dataset.demand(start_pos+1:end_pos)/1000)
hold on
plot([start_pos+1: end_pos], demand_pred/1000)
xlim([start_pos-30, end_pos+30])
grid on

% Plot autocorrelation and partial autocorrelation of the residuals
residuals_plt = dataset.std_demand(start_pos+1:end_pos) - y_GLM_train;

figure()
autocorr(residuals_plt, 'NumLags', 50);
xlabel('Days')
xlim([-2, 52])
ylim([-0.2, 1.2])

figure()
parcorr(residuals_plt, 'NumLags', 50);
xlabel('Days')
xlim([-2, 52])
ylim([-0.4, 1.2])

%% NAX Model
% rng(100)

% Needed data stored in a Table
calendar_var_NAX = array2table(regressors);
dataset_NAX = calendar_var_NAX(start_pos+1:val_pos, :);
dataset_NAX.drybulb = dataset.drybulb(start_pos+1:val_pos);
dataset_NAX.dewpnt = dataset.dewpnt(start_pos+1:val_pos);
dataset_NAX.std_demand = dataset.std_demand(start_pos+1:val_pos);
dataset_NAX.demand = dataset.demand(start_pos+1:val_pos);
dataset_NAX.residuals = residuals;

% Hyperparameters
LIST_HIDDEN_NEURONS = [3, 4, 5, 6];
LIST_ACT_FUN = ["softmax"; "logsig"];
LIST_LEARN_RATE = [0.1, 0.01, 0.003, 0.001];
LIST_BATCH_SIZE = [50,5000];
LIST_REG_PARAM = [0.001, 0.0001, 0];

% LOSS_FUN = "mll"; % ONLY RUN MLL IF MLL IS INSTALLED IN THE PC
LOSS_FUN = "mse";

[hidden_neurons, act_fun, lrn_rate, reg_param, batch_size,min_RMSE, all_RMSE] = ...
    find_hyperparam(dataset_NAX, LOSS_FUN,...
    LIST_HIDDEN_NEURONS, LIST_ACT_FUN, LIST_LEARN_RATE,LIST_REG_PARAM,LIST_BATCH_SIZE, M, m, start_date, end_date, val_date);

disp("hidden_neurons: "+string(hidden_neurons)  + " - act_fun: "+string(act_fun)...
    + " - lrn_rate: "+string(lrn_rate)  + " - reg_param: "+string(reg_param)...
    +"batch_size: "+ batch_size)

disp("RMSE: "+string(min_RMSE))

%% Parameters calibration and Confidence interval on the test set. Train set: 2009 - 2011. Test set: 2012 

start_date = 2009;
end_date   = 2011;
test_date  = 2012;
start_pos = (start_date -2008)*365;
end_pos   = (end_date+1 -2008)*365;
test_pos  = (test_date+1 -2008)*365;

[y_GLM_test, y_GLM_train, sigma_GLM] = GLM(dataset, regressors, start_date, end_date, test_date); %predicted values
y_GLM = [y_GLM_train; y_GLM_test];
residuals = dataset.std_demand(start_pos+1:test_pos) - y_GLM;

calendar_var_NAX = array2table(regressors);
dataset_NAX = calendar_var_NAX(start_pos+1:test_pos, :);
dataset_NAX.drybulb = dataset.drybulb(start_pos+1:test_pos);
dataset_NAX.dewpnt = dataset.dewpnt(start_pos+1:test_pos);
dataset_NAX.std_demand = dataset.std_demand(start_pos+1:test_pos);
dataset_NAX.residuals = residuals;

[mu_NAX, sigma_NAX] = NAX(dataset_NAX, LOSS_FUN, hidden_neurons, act_fun, lrn_rate, reg_param,batch_size, start_date, end_date, test_date, 1);

y_NAX_test = mu_NAX' + y_GLM_test;

%% Confidence interval

[y_NAX_l, y_NAX_u] = ConfidenceInterval(y_NAX_test, sigma_NAX, 0.95, M, m);
estimated_values = destd(y_NAX_test, M, m);
 
figure()
hold on
x = 1:length(estimated_values);
x_area = [x, fliplr(x)];
y_area = [y_NAX_l', fliplr(y_NAX_u')];
fill(x_area, y_area, [1 0.7 0.7]);
plot(dataset.demand(end_pos+1:test_pos), 'b', 'LineWidth', 1.2) % prendere colonna demand
plot(y_NAX_l, 'r', 'LineWidth', 0.4)
plot(y_NAX_u, 'r', 'LineWidth', 0.4)
plot(estimated_values, 'r', 'LineWidth', 0.8)
hold off
%% FOR 
RMSE = zeros(5,3)
MAPE = zeros(5,3)
APL = zeros(5,3)
for i = [0:4] %0:4
    start_date = 2009+i;
    end_date   = 2011+i;
    test_date  = 2012+i

    start_pos = (start_date - 2008)*365;
    end_pos   = (end_date+1 - 2008)*365;
    test_pos   = (test_date+1 - 2008)*365;
    
    [y_GLM_test, y_GLM_train, sigma_GLM] = GLM(dataset, regressors, start_date, end_date, test_date); % predicted values on validation set and train set
                                                                                                      % and standard deviation of the fitted model
    y_GLM = [y_GLM_train; y_GLM_test];
    residuals = dataset.std_demand(start_pos+1:test_pos) - y_GLM;
    
    y = dataset.demand(end_pos+1:test_pos);
    RMSE_GLM = rmse(y, destd(y_GLM_test, M, m));
    MAPE_GLM = mape(y, destd(y_GLM_test, M, m));
    
    calendar_var_NAX = array2table(regressors);
    dataset_NAX = calendar_var_NAX(start_pos+1:test_pos, :);
    dataset_NAX.drybulb = dataset.drybulb(start_pos+1:test_pos);
    dataset_NAX.dewpnt = dataset.dewpnt(start_pos+1:test_pos);
    dataset_NAX.std_demand = dataset.std_demand(start_pos+1:test_pos);
    dataset_NAX.residuals = residuals;
    
    % NAX model
    [mu_NAX, sigma_NAX] = NAX(dataset_NAX, LOSS_FUN, hidden_neurons, act_fun, lrn_rate, reg_param,batch_size, start_date, end_date, test_date, 1);
    y_NAX_test = mu_NAX' + y_GLM_test;

    [y_NAX_l, y_NAX_u] = ConfidenceInterval(y_NAX_test, sigma_NAX, 0.95, M, m);
    estimated_values = destd(y_NAX_test, M, m);
    
    figure()
    hold on
    x = 1:length(estimated_values);
    x_area = [x, fliplr(x)];
    y_area = [y_NAX_l', fliplr(y_NAX_u')];
    fill(x_area, y_area, [1 0.7 0.7]);
    plot(dataset.demand(end_pos+1:test_pos), 'b', 'LineWidth', 1.2) % prendere colonna demand
    plot(y_NAX_l, 'r', 'LineWidth', 0.4)
    plot(y_NAX_u, 'r', 'LineWidth', 0.4)
    plot(estimated_values, 'r', 'LineWidth', 0.8)
    
    RMSE_NAX = rmse(y, destd(y_NAX_test, M, m));
    MAPE_NAX = mape(y, destd(y_NAX_test, M, m));
    
    % ARX Model
    [y_ARX_test, sigma_ARX] = ARX(dataset_NAX, start_date, end_date, test_date);
    
    RMSE_ARX = rmse(y, destd(y_ARX_test, M, m));
    MAPE_ARX = mape(y, destd(y_ARX_test, M, m));
   
    figure()
    plot(y)
    hold on
    plot(destd(y_ARX_test, M, m))
    
    
    % Pinball Loss
    pinball_values_GLM = pinball(y, y_GLM_test, sigma_GLM, M, m);   % y_pred = output of GLM (prediction of std(log_demand))
    pinball_values_NAX = pinball(y, y_NAX_test, sigma_NAX, M, m);
    pinball_values_ARX = pinball(y, y_ARX_test, sigma_ARX, M, m);

    % Pinball Loss Graph
    figure()
	plot([1:length(pinball_values_GLM)]/100, pinball_values_GLM/1000, 'r--', ...
         [1:length(pinball_values_ARX)]/100, pinball_values_ARX/1000, 'b:', ...
         [1:length(pinball_values_NAX)]/100, pinball_values_NAX/1000, 'k')
    legend('GLM', 'ARX', 'NAX', 'Location', 'NorthEast')
    xlabel('Quantile')
    ylabel('Pinball Loss [GWh]')
    
    
    % Backtest
    confidence_levels = [0.9:0.01:1];
    [backtested_levels_GLM, LR_Unc_GLM, LR_Cov_GLM] = backtest(y, y_GLM_test, confidence_levels, sigma_GLM, M, m);
    [backtested_levels_NAX, LR_Unc_NAX, LR_Cov_NAX] = backtest(y, y_NAX_test, confidence_levels, sigma_NAX, M, m);
    [backtested_levels_ARX, LR_Unc_ARX, LR_Cov_ARX] = backtest(y, y_ARX_test, confidence_levels, sigma_ARX, M, m);
 
    disp('LR_GLM')
    LR_Unc_GLM
    LR_Cov_GLM
    disp('LR_NAX')
    LR_Unc_NAX
    LR_Cov_NAX
    disp('LR_ARX')
    LR_Unc_ARX
    LR_Cov_ARX

    figure()
    plot(confidence_levels, backtested_levels_GLM, 'r--', ...
         confidence_levels, backtested_levels_ARX, 'b:', ...
         confidence_levels, backtested_levels_NAX, 'k', ...
         confidence_levels, confidence_levels, 'c.', 'MarkerSize', 8)
    legend('GLM', 'ARX', 'NAX', 'Nominal Level', 'Location', 'NorthWest')
	xlabel('Nominal Level \alpha')
    ylabel('Backtested Level')
    xlim([0.895 1.005])
    i=i+1
    RMSE(i,1) = RMSE_GLM;
    RMSE(i,2) = RMSE_ARX;
    RMSE(i,3) = RMSE_NAX;
    MAPE(i,1) = MAPE_GLM;
    MAPE(i,2) = MAPE_ARX;
    MAPE(i,3) = MAPE_NAX;
    APL(i,1) = mean(pinball_values_GLM);
    APL(i,2) = mean(pinball_values_ARX);
    APL(i,3) = mean(pinball_values_NAX);
    
    
end

