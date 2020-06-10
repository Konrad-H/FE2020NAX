function [y_pred_test, y_pred_train, sigma] = GLM(df, regressors, first_year, last_year, test_year)
%[y_GLM_val, y_GLM_train, sigma] = GLM(dataset, regressors, start_date, end_date, val_date);

% This function implements the GLM
%
% INPUTS:
% df:           DataFrame containing variable std_demand: response to be predicted
% regressors:   DataFrame containing the regressors of the GLM model to be fitted
% first_year:   Year from where the train set starts at 1/1
% last_year:    Year in which the train set ends at 31/12
% test_year:    Year of the test_set from 1/1 to 31/12
%
% OUTPUTS:
% y_pred_test:  Predicted response on test set
% y_pred_train: Predicted response on train set
% sigma:        Standard Deviation of the fitted model

last_year = last_year + 1;
test_year = test_year + 1;

firsty_pos = (first_year - 2008)*365;
lasty_pos = (last_year - 2008)*365;
testy_pos = (test_year - 2008)*365;

% Build GLM model
X_train = regressors(firsty_pos+1:lasty_pos, :);
y_train = df.std_demand(firsty_pos+1:lasty_pos); % 1 -> std_demand   DA CAMBIARE

% Linear Regression 
linear_model = fitlm(X_train, y_train)

% Get std. dev. of the model
y_pred_train = predict(linear_model, X_train);
residuals = y_train - y_pred_train;
sigma = sqrt(residuals'*residuals/(length(residuals)-1));

% Prediction on test set
X_test = regressors(lasty_pos+1:testy_pos, :);
y_pred_test = predict(linear_model, X_test);

end
