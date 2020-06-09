function [pinball_values] = pinball(y, y_test, sigma, M, m)

% This function computes Pinball Loss for every quantile
%
% INPUTS:
% y:        vector of the realised demand on the test set
% y_test:   vector of mean values of the predicted standardized log_demand
% sigma:    vector of standard deviations of the predicted standardized log_demand
% M:        maximum of the log_demand observed over the years: 2008 - 2016
% m:        minimum of the log_demand observed over the years: 2008 - 2016
%
% OUTPUT:
% pinball_values:   vector of Pinball Loss for every quantile


alpha = 0.01:0.01:0.99;                                                    % Confidence Levels
gauss_quant = norminv(alpha);                                              % Gaussian Quantile
quant = exp((((M-m)*y_test) + m) + sigma*(M-m)*gauss_quant);               % Quantiles of lognormal demand distribution
NIP = abs(quant-y).*((quant>y).*(1-alpha) + (y>=quant).*alpha);
pinball_values = mean(NIP);                                                % Pinball Loss

end