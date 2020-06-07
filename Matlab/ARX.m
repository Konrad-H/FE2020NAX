function [y_pred_test, sigma] = ARX (df, first_year, last_year, test_year)

% last year is intendeed as 31/12 of last year
last_year = last_year + 1;
test_year = test_year + 1;

firsty_pos = (first_year - first_year)*365;
lasty_pos = (last_year - first_year)*365;
testy_pos = (test_year - first_year)*365;

% Build ARX model
X = df(:,[1:10]); %take drybulb, dewpnt, all regressors except intercept's one % DA GUARDARE!!
y = df(:,11); % take std_demand
X_train = X([firsty_pos+1:lasty_pos],:);
y_train = y([firsty_pos+1:lasty_pos]);

data_train = iddata(y_train, X_train);
sys = arx(data_train, [1 ones(1,size(X_train,2)) zeros(1,size(X_train,2))])

sigma = sqrt(sys.Report.Fit.MSE);

X_test = X(lasty_pos+1:testy_pos,:);
data_test = iddata(X_test);
prediction = forecast(sys, data_train, size(X_test,1), data_test);
y_pred_test = prediction.y;


end