function df_std = data_standardize(dataset)

% This function standardizes weather variables of dataset, mapping them in [0,1], 
% moreover it adds log_demand and standardize log_demand to the table
%
% INPUT:
% dataset:  table containing energy demand, weather variables and calendar variables
%
% OUTPUT:
% df_std:   table containing energy demand, energy log_demand, standardized energy log_demand, 
%           standardized weather variables and calendar variables

%% Load dataset
df_std = dataset;

%% Standardize weather variables
df_std.drybulb = standardize(df_std.drybulb);
df_std.dewpnt = standardize(df_std.dewpnt);

%% Add variable log_demand
log_demand = log(df_std.demand);
df_std.log_demand = log_demand;
    
%% Add variable standardized log_demand
df_std.std_demand = standardize(log_demand);

end