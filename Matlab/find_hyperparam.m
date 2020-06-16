function [hidden_neurons, act_fun, lrn_rate, reg_param, min_RMSE, all_RMSE] = find_hyperparam ...
(dataset, loss_f, list_hidden_neurons, list_act_fun, list_lrn_rate,list_reg_param, M, m, first_year, last_year, test_year)

%
% INPUTS:
% dataset:              Table containing useful data
% list_hidden_neurons:  considered number of neurons in the hidden layer
% list_act_fun:         considered activation function of the hidden layer
% list_reg_param:       considered parameter of regularization
% M:                    Maximum of the log_demand
% m:                    Minimum of the log_demand
% frist year:           Year of beginning of training set
% last_year:            Year of end of the training set
% test_year:            Year of the test set
%
% OUTPUTS:
% hidden neurons:       optimal number of neurons in the hidden layer
% act_fun:              optimal activation function of the hidden layer
% reg_param:            optimal parameter of the regularization
% min_RMSE              minimum Root Mean Squared Error (RMSE)
% all_RMSE              3D matrix containing RMSE of each combination
%

% position of test set
lasty_pos = (last_year+1 - first_year)*365;
testy_pos = (test_year+1 - first_year)*365;

% Real and GLM demand
y = dataset.demand(lasty_pos+1:testy_pos);
y_GLM = dataset.std_demand(lasty_pos+1:testy_pos) - dataset.residuals(lasty_pos+1:testy_pos);

L1 = length(list_hidden_neurons);
L2 = length(list_act_fun);
L3 = length(list_lrn_rate);
L4 = length(list_reg_param);
all_RMSE = zeros(L1, L2, L3,L4);

% for loop to try all possible combinations
for n1 = [1:L1]
    for n2 =[1:L2] 
        for n3 =[1:L3] 
            for n4 = [1:L4]
                [mu_NAX, ~] = NAX(dataset, loss_f, list_hidden_neurons(n1), ...
                    list_act_fun(n2,:), list_lrn_rate(n3),list_reg_param(n4),...
                    first_year, last_year, test_year, 0); % predicted residuals
                % demand predicted by the neural network
                if loss_f=='mll'
                    mu_NAX = mu_NAX(:,1);
                end
                y_NAX = y_GLM' + mu_NAX; 
                RMSE = rmse(y, destd(y_NAX',M,m));

            end
        end
    end
end

% Find the minimum RMSE and relative optimal parameters
[~, idx] = min(all_RMSE(:));            
[n1, n2, n3,n4] = ind2sub(size(all_RMSE),idx)
min_RMSE = all_RMSE(n1,n2,n3,n4);
hidden_neurons = list_hidden_neurons(n1);
act_fun = list_act_fun(n2,:);
lrn_rate = list_lrn_rate(n3);
reg_param = list_reg_param(n4);

end

            