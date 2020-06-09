clear all
clc

load regressors
load residuals
regressors = cell2mat(regressors);
regressors = (regressors-mean(regressors)) ./ std(regressors);
regressor_input = mat2cell([regressors(1:end-1,:)'; regressors(2:end,:)'],ones(1,2)*10,ones(1,365*8 - 1));
residuals = residuals';

regressors_train = regressor_input(:,1:365*3 -1);
residuals_train = residuals(:,2:365*3);

regressors_test = regressor_input(:,365*3:365*4-1);
residuals_test = residuals(:,365*3+1:365*4);

% layrecnet(layerDelays,hiddenSizes,trainFcn)
% net = layrecnet(1,3);
% net.numInputs = 10;
% net.InputConnect = [ones(1,10); zeros(1,10)];
% net.layers{1}.transferFcn = 'softmax';   % ACT_FUN
% net.performFcn = 'mse';

net = layrecnet(1,3);
net.numInputs = 2;
net.inputs{1}.size = 10;
net.inputs{2}.size = 10;
net.InputConnect = [ones(1,2); zeros(1,2)];
net.layers{1}.transferFcn = 'softmax';   % ACT_FUN
net.performFcn = 'mse';

[Xs,Xi,Ai,Ts] = preparets(net,regressors_train,residuals_train);

net = train(net, Xs, Ts, Xi, Ai);

residuals_pred = net(cell2mat(regressors_test))

figure
plot([1:length(residuals_pred)],residuals_pred)
hold on
plot([1:length(residuals_pred)],cell2mat(residuals_test))

