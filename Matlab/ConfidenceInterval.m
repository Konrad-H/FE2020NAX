function [IC_l, IC_u] = ConfidenceInterval(mu, sigma, confidence_level, M, m)

% This function computes lower and upper bound of a confidence interval at confidence_level for each element y_t
% of a vector y, with standardized log(y_t) distributed as N(mu_t, sigma_t^2)
% 
% INPUTS:
% mu:       vector of mean values
% sigma:    vector of standard deviations
% confidence_level
% M:        maximum of the log_demand observed over the years: 2008 - 2016
% m:        minimum of the log_demand observed over the years: 2008 - 2016
%
% OUTPUTS:
% IC_l:     vector of lower bounds
% IC_u:     vector of upper bounds

alpha_l = (1-confidence_level)/2;
alpha_u = 1-alpha_l;

% lower bound
gauss_quant_l = norminv(alpha_l);
IC_l = exp((((M-m)*mu) + m) + (M-m)*sigma.*gauss_quant_l);

% upper bound
gauss_quant_u = norminv(alpha_u);
IC_u = exp((((M-m)*mu) + m) + gauss_quant_u*sigma*(M-m));

end
