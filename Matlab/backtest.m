function [backtested_levels, LR_Uncond, LR_Cond] = backtest(y_real, y_pred, confidence_levels, sigma, M, m)

% This function provides backtested_levels, Unconditional Coverage Likelihood Ratio and
% Conditional Coverage Likelihood Ratio
%
% INPUTS
% y_real:   vector of the realised demand on the test set
% y_pred:   vector of mean values of the predicted standardized log_demand
% confidence_levels:    vector of confidence levels
% sigma:    vector of standard deviations of the predicted standardized log_demand
% M:        maximum of the log_demand observed over the years: 2008 - 2016
% m:        minimum of the log_demand observed over the years: 2008 - 2016
% 
% OUTPUTS:
% backtested_levels/N:  fraction of realised demand falling inside a confidence interval
% LR_Uncond:    Unconditional Coverage Likelihood Ratio
% LR_Cond:      Conditional Coverage Likelihood Ratio

[IC_l, IC_u] = ConfidenceInterval(y_pred, sigma, confidence_levels, M, m); % Confidence Interval for each Confidence level
exceptions = (y_real < IC_l) + (y_real > IC_u);                            % It takes value 1 when the realisation falls out of the CI

% backtested levels
backtested_levels = 1 - mean(exceptions);

% LR: Unconditional Covarage Test at 95%
alpha = 0.05;
exception_95 = exceptions(:,6);                                            % Exceptions vector at confidence level 95%
exception_number = sum(exception_95);                                      % Number of exceptions
N = length(exception_95);                                                  % Total number of considered realisations
alpha_hat = exception_number/N;                                            % Backtested significance level
alpha_hat = mean(exception_95);                                            % Theoretical significance level
LR_Uncond = -2*log((alpha/alpha_hat)^exception_number*((1-alpha)/(1-alpha_hat))^(N-exception_number));

% LR: Conditional Covarage Test
% formato vettoriale??
% Number of values falling in or out of the CI, conditioned to the previous outcome 
N_11 = exception_95'*[0; exception_95(1:end-1)];
N_01 = exception_number - N_11;
N_00 = (1-exception_95)'*[1; (1-exception_95(1:end-1))];
N_10 = N - exception_number - N_00;

% Backtested significace levels
alpha_01 = N_01/(N_00 + N_01);
alpha_11 = N_11/(N_11 + N_10);

LR_num = alpha^exception_number*(1-alpha)^(N-exception_number);
LR_den = alpha_01^N_01*(1-alpha_01)^N_00*alpha_11^N_11*(1-alpha_11)^N_10;
LR_Cond = -2*log(LR_num/LR_den);

end