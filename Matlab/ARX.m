function [y_pred_test, sigma] = ARX(df, first_year, last_year, test_year)

% This function implements an ARX model
%
% INPUTS:
% df:           DataFrame containing varible std_demand, i.e. response to be predicted, 
%                and the exogenous regressors: calendar variables and weather conditions
% first_year:   Year from where the train set starts at 1/1
% last_year:    Year in which the train set ends at 31/12
% test_year:    Year of the test_set from 1/1 to 31/12
%
% OUTPUTS:
% y_pred_test:  Predicted response on test set
% sigma:        Standard Deviation of the fitted model

% last year is intendeed as 31/12 of last year
last_year = last_year + 1;
test_year = test_year + 1;

firsty_pos = (first_year - first_year)*365;
lasty_pos = (last_year - first_year)*365;
testy_pos = (test_year - first_year)*365;

% Build ARX model
% Train set is defined
X = df(:, 1:10); %take drybulb, dewpnt, all regressors except intercept's one % DA GUARDARE!!
y = df(:, 11); % take std_demand
X_train = X(firsty_pos+1:lasty_pos, :);
y_train = y(firsty_pos+1:lasty_pos, :);
X_train = table2array(X_train);
y_train = table2array(y_train);

% ARX model is defined 
data_train = iddata(y_train, X_train);
sys = arx(data_train, [1 ones(1,size(X_train,2)) zeros(1,size(X_train,2))]);

% Model Standard Deviation
sigma = sqrt(sys.Report.Fit.MSE);

% Prediction on test set
X_test = X(lasty_pos+1:testy_pos, :);
X_test = table2array(X_test);
data_test = iddata(X_test);
prediction = forecast(sys, data_train, size(X_test,1), data_test);
y_pred_test = prediction.y;

end