clear all
clc

load regressors
load residuals

regressors_train = regressors(1:365*3,:);
residuals_train = residuals(1:365*3,:);

regressors_test = regressors(365*3+1:365*4,:);
residuals_test = residuals(365*3+1:365*4,:);

validation_data = {cell2mat(regressors_test), cell2mat(residuals_test)}';
options = trainingOptions('adam', 'ValidationData', validation_data);   % used by trainNetwork

% layrecnet(layerDelays,hiddenSizes,trainFcn)
net = layrecnet(1,3);
net.numInputs = 10;
net.InputConnect = [ones(1,10); zeros(1,10)];
net.layers{1}.transferFcn = 'softmax';   % ACT_FUN

[Xs,Xi,Ai,Ts] = preparets(net,regressors_train',residuals_train');
net = train(net, Xs, Ts, Xi, Ai);

residuals_pred = net(regressors_test')