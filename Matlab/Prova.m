clear all
clc

% layrecnet(layerDelays,hiddenSizes,trainFcn)
net = layrecnet(1, 3)
% net.numInputs vs. net.inputs{i}.size
net.layers{1}.size = 3  % HIDDEN_NEURONS
net.layers{2}.size = 2
net.layers{1}.transferFcn = 'softmax'   % ACT_FUN
net.layers{2}.transferFcn = 'poslin'
net.numInputs = 10
net.InputConnect = [ones(1,10); zeros(1,10)]
view(net)

load residuals
load regressors
residuals_train = residuals(1:365*3,:);
regressors_train = regressors(1:365*3,:);
residuals_test = residuals(365*3+1:365*4,:);
regressors_test = regressors(365*3+1:365*4,:);
X_train = [regressors_train residuals_train];
X_test = [regressors_test residuals_test];

[Xs, Xi, Ai, Ts, EWs, shift] = preparets(net, regressors_train', {}, {})

% [Xs, Xi, Ai, Ts, EWs, shift] = preparets(net, Xnf, Tnf, Tf, EW)
% Xnf = non-feedback inputs, Tnf = non-feedback targets
% Tf = feedback targets, EW = error weigths
% [Xs, Xi, Ai, Ts, EWs, shift] = preparets(net, X, {}, T);

% net = train(net, Xs, Ts, Xi, Ai);