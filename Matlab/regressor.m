function regressors = regressor(df_std)

% This function builds the regressors matrix for GLM 
%
% INPUT:
% df_std:   table containing the variables needed to build the regressors
%
% OUTPUT:
% regressors:   regressors matrix for GLM

N_data = length(df_std.years);
omega = 2*pi/365;   % frequence for yearly and semestral trend

% create dummy variables for Saturdays and Sundays
Sat = strcmp(df_std.day_of_week, "Sat");
Sun = strcmp(df_std.day_of_week, "Sun");

time_in_days = 0:(N_data-1);
% first day is 1/3/2008, lacking 59 calendar days, as we start to count from the first day in the original dataframe
time_since_dataset = (2008-2003)*365-59;
t = ((time_in_days) + time_since_dataset)';
time = t/max(t);    % reparametrization of time

%% Covariates are defined
regressors = [time,...
    sin(omega*t), cos(omega*t), sin(2*omega*t), cos(2*omega*t),...
    Sat, Sun, df_std.holiday]; 

end