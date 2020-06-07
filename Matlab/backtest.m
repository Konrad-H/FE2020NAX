function [backtested_levels, LR_Uncond, LR_Cond] = backtest(y_real, y_pred, confidence_levels, sigma, M, m)


[IC_l, IC_u] = ConfidenceInterval(y_pred, sigma, confidence_levels, M, m);
exceptions = (y_real < IC_l) + (y_real > IC_u);

% backtested levels
backtested_levels = 1 - mean(exceptions)

% LR: Unconditional Covarage Test at 95%
alpha = 0.05;
exception_95 = exceptions(:,6)
exception_number = sum(exception_95);
N = length(exception_95);
alpha_hat = exception_number/N;
alpha_hat = mean(exception_95);
LR_Uncond = -2*log((alpha/alpha_hat)^exception_number*((1-alpha)/(1-alpha_hat))^(N-exception_number));

% LR: Conditional Covarage Test
% formato vettoriale??
N_11 = exception_95'*[0; exception_95(1:end-1)]
N_01 = exception_number - N_11
N_00 = (1-exception_95)'*[1; (1-exception_95(1:end-1))]
N_10 = N - exception_number - N_00

alpha_01 = N_01/(N_00 + N_01);
alpha_11 = N_11/(N_11 + N_10);

LR_num = alpha^exception_number*(1-alpha)^(N-exception_number);
LR_den = alpha_01^N_01*(1-alpha_01)^N_00*alpha_11^N_11*(1-alpha_11)^N_10;

LR_Cond = -2*log(LR_num/LR_den);

end