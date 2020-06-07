clear all
clc

load regressors
load residuals

regressors_train = regressors(1:365*3,:);
residuals_train = cell2mat(residuals(1:365*3,:));

regressors_test = cell2mat(regressors(365*3+1:365*4,:));
residuals_test = cell2mat(residuals(365*3+1:365*4,:));

validation_data = {regressors_test, residuals_test}';
options = trainingOptions('adam', ...
          'Verbose', 1, ...
          'MaxEpochs', 600, ...
          'MiniBatchSize', 50, ...
          'Shuffle', 'never', ...
          'ValidationData', validation_data, ...
          'ValidationPatience', 10, ...
          'InitialLearnRate', 0.003, ...
          'L2Regularization', 1e-4);   % used by trainNetwork

input_layer = sequenceInputLayer(10);
hidden_layer = lstmLayer(3);
output_layer = regressionLayer;
layers = [input_layer hidden_layer output_layer]

trainedNet = trainNetwork(cell2mat(regressors_train)', cell2mat(residuals_train)', layers, options)
