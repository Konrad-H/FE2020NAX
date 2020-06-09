function vector = destd(std_vector, M, m)
%
% INPUTS:
% vector:   vector to be destandardized
% M:        maximum log_demand value
% m:        minimum log_demand value
%
% OUTPUT:
% vector:   destandardized vactor: it represent the demand in the original domain

log_vec = std_vector*(M-m) + m;
vector = exp(log_vec);

end
