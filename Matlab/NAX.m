function [targets_pred, sigma]= NAX(dataset, loss_f ,hidden_neurons, act_fun, lrn_rate, reg_param, first_year, last_year, test_year, flag)

% This function builds and calibrates the NAX network, with loss function
% Mean Squared Error (MSE), and computes standard deviation of the fitted
% model and a prediction on the test set
%
% INPUTS:
% dataset:          table containing inputs and target of the Neural Network 
% hidden neurons:   number of neurons in the hidden layer
% act_fun:          activation function of the hidden layer
% reg_param:        parameter of the regularization
% frist year:       year of beginning of training set
% last_year:        year of end of the training set
% test_year:        year of the test set
%
% OUTPUTS:
% targets_pred:     predicted target on the test set
% sigma:            standard deviation of the model
%

% position of train and test set
last_year = last_year + 1;
test_year = test_year + 1;
firsty_pos = (first_year - first_year)*365;
lasty_pos = (last_year - first_year)*365;
testy_pos = (test_year - first_year)*365;

% definition and preprocessing of network features and targets
regressors = table2array(dataset(:, 1:10));
targets = dataset.residuals;

regressors = (regressors-mean(regressors)) ./ std(regressors);

inputs = 10;

if inputs==20
    regressors_input = mat2cell([regressors(1:end-1,:)'; regressors(2:end,:)'],20,ones(1,testy_pos - 1)); %regressors [x(t), x(t-1)]
else
    regressors_input = mat2cell(regressors',10,ones(1,testy_pos));
end
targets = targets';

% divsion of the data in train and test set
if inputs==20
    regressors_train = regressors_input(:,firsty_pos+1:lasty_pos-1);
    targets_train = targets(firsty_pos+2:lasty_pos);
    regressors_test = regressors_input(:,lasty_pos:testy_pos-1);
    targets_test = targets(lasty_pos+1:testy_pos);
else
    regressors_test = regressors_input(:,lasty_pos+1:testy_pos);
    targets_test = targets(lasty_pos+1:testy_pos);
    regressors_train = regressors_input(:,firsty_pos+1:lasty_pos);
    targets_train = targets(firsty_pos+1:lasty_pos);
end

% Building of the NAX network
net = narxnet(0,1,hidden_neurons, 'closed');%0: time lag on input
                                            %1: time lag on autoregresion
                                            %number of hidden neurons
                                            %closed loop network
net.inputs{1}.size = inputs;            %10 inputs x(t), 10 inputs x(t-1)
net.InputConnect = [1; 0];          %connection of input to hidden layer
net.layers{1}.transferFcn = act_fun;
net.performFcn = loss_f;
net.trainParam.lr = lrn_rate;
if loss_f == "mll"
    net.trainFcn = 'trainrp';%loss function: MSE
    targets_train=[targets_train;targets_train]
    targets_test=[targets_test;targets_test]
end
net.performParam.regularization = reg_param;

% Training of the neural network
net.trainParam.showWindow = flag;
if inputs==20
    delta=1 ;
else
    delta=0;
end    
targets_train = mat2cell(targets_train, size(targets_train,1), ones(1,lasty_pos-delta));
net = train(net,regressors_train,targets_train);

% Standard Deviation of the calibrated network
train_pred = (net(cell2mat(regressors_train)));
err = train_pred(1,:) - cell2mat(targets_train(1,:));
sigma = sqrt(err*err'/(length(err)-1));

% Prediction on test_set
targets_pred = net(cell2mat(regressors_test));

% Plot: theoretical residuals vs. predicted residuals
if (flag)
    plot([1:length(targets_pred)],targets_pred(1,:) )
    hold on
    plot([1:length(targets_pred)],targets_test(1,:) )
end
