function regressors = regressor(df_std)


N_data = length(df_std.years);
omega = 2*pi/365;

%%
Sat = strcmp(df_std.day_of_week,"Sat");
Sun = strcmp(df_std.day_of_week,"Sun");

time_in_days = 0:(N_data-1);

%% FIRST DAY IS march 1st, lacking 59 calendar days
time_since_dataset = (2008-2003)*365-59;

t=( (time_in_days)+time_since_dataset )';
time = t/max(t); % reparametrization of time

%% COVARIATES
regressors =[time,...
    sin(omega*t), cos(omega*t), sin(2*omega*t), cos(2*omega*t),...
    Sat, Sun, df_std.holiday]; 


end