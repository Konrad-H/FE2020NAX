function perfs = apply(t,y,e,param)
%MSE.APPLY Calculate performances

% Copyright 2012-2015 The MathWorks, Inc.
  strike = 1;
  var = y(2,:);
  mu = y(1,:);
  var = max(var, strike);
  K = 1;
  perfs = K*(log(var)+(mu-t).^2./(2*var));
end
