function err = mape(predictions, targets)

% This function computes the Mean Absolute Percentage Error (MAPE)
%
% INPUT:
% predictions:  predicted values
% targets:      real values
%
% OUTPUT: 
% err:          Mean Absolute Percentage Error

err = mean(abs((predictions - targets)./targets));

end
