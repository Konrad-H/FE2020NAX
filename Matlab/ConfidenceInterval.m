function [IC_l, IC_u] = ConfidenceInterval(mu, sigma, confidence_level, M, m)

alpha_l = (1-confidence_level)/2;
alpha_u = 1-alpha_l;

% lower bound
gauss_quant_l = norminv(alpha_l);
IC_l = exp((((M-m)*mu) + m) + (M-m)*sigma.*gauss_quant_l);
% upper bound
gauss_quant_u = norminv(alpha_u);
IC_u = exp((((M-m)*mu) + m) + gauss_quant_u*sigma*(M-m));

end
