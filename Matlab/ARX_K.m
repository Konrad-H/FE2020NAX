clear all
clc

load regressors
load residuals
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

%% SIMPLE AR SYSTEM
data_train = iddata(y_train);
sys = ar(data_train,1);

data_val = iddata(y_val);
compare(data_val, sys, 1);

%% ARX MISO
na = 1;
nb = 1*ones(1,10);
nk = 0*ones(1,10);

data_train = iddata(y_train, x_train,1 );
sys = arx(data_train, 'na',na, 'nb',nb,'nk', nk);

data_val = iddata(y_val, x_val );
%compare(data_val, sys, 1);

y_pred = predict(sys, data_val,1);


hold on
t=1:364

plot(t,y_val(1:364),t,y_pred.OutputData(2:365), '.')
hold off