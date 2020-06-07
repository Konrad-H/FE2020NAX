function [pinball_values] = pinball(y, y_test, sigma, M, m)

% y_test e y col. vector
alpha = 0.01:0.01:0.99;
gauss_quant = norminv(alpha);
quant = exp((((M-m)*y_test) + m) + sigma*(M-m)*gauss_quant);
NIP = abs(quant-y).*((quant>y).*(1-alpha) + (y>=quant).*alpha);
size(NIP)
pinball_values = mean(NIP);

end