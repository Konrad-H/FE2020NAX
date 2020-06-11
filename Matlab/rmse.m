function err = rmse(predictions, targets)

% This function computes the Root Mean Squared Error (RMSE)
%
% INPUT:
% predictions:  predicted values
% targets:      real values
%
% OUTPUT: 
% err:          Root Mean Squared Error

residuals = predictions - targets;
err = sqrt(residuals'*residuals/length(residuals));

end
