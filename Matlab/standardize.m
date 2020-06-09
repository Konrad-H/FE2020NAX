function vector = standardize(vector)
%
% INPUTS:
% vector:   vector to be standardized
%
% OUTPUT:
% vector:   de\standardized vactor

M=max(vector);
m=min(vector);
vector = (vector-m)/(M- m);
end
