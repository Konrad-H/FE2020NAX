clear all
clc

load regressors
load residuals
load demand
%%
y = cell2mat(residuals);
x = cell2mat(regressors);

start_pos=365*0;
train_split = (start_pos) +365*3;
end_pos = train_split + 365;

y_train = y( (1+start_pos) : train_split);
x_train = x( (1+start_pos) : train_split,:);

y_val = y( (1+train_split) : end_pos);
x_val = x( (1+train_split) : end_pos,:);

%% ARX MISO
na = 1;
nb = 1*ones(1,10);
nk = 0*ones(1,10);

data_train = iddata(y_train, x_train,1 );
sys = arx(data_train, 'na',na, 'nb',nb,'nk', nk);

data_val = iddata(y_val, x_val );
%compare(data_val, sys, 1);

y_pred = predict(sys, data_val,1);
%%
M = max(log(demand));
m = min(log(demand));
std_demand = (log(demand)-m)/(M-m);

de_std = @(x) exp(x*(M-m)+m);
X0Obj = idpar(y_train(end));
options = forecastOptions('InitialCondition',X0Obj);
y_pred = forecast(sys, data_train, size(x_val,1), x_val, options);

t=1:365;
demand_real = std_demand((1+train_split) : end_pos);
demand_GLM  = demand_real - y_val;
demand_ARX  = demand_GLM + y_pred.y;

demand_real =de_std(demand_real);
demand_GLM =de_std(demand_GLM);
demand_ARX =de_std(demand_ARX);

rmse(demand_ARX,demand_real)
plot(t,demand_real,t,demand_GLM,'y',t,demand_ARX,'r');

