function  df_std=data_standardize(dataset)

%% LOAD
df_std = dataset;
    
%% add
log_demand = log(df_std.demand);
df_std.log_demand = log_demand;
df_std.std_demand = standardize(log_demand);
df_std.drybulb = standardize(df_std.drybulb);
df_std.dewpnt = standardize(df_std.dewpnt);

end