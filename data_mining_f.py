import pandas as pd
import numpy as np
from standard_and_error_functions import standardize

def data_mining(PATH):
    
    # This function builds a dataframe with energy demand, weather variables and calendar variables
    #
    # INPUT:
    # PATH:     path of the original dataset gefcom.csv
    #
    # OUTPUT:
    # ready_df: dataframe containing energy demand, weather variables and calendar variables
        
    # Load original dataset
    df = pd.read_csv(PATH)
    # df.head()

    # Filter data
    df_cut = df[(df['zone']=='TOTAL') &     # consider the whole New England area
                (df['year']>=2008) & (df['year']<=2016) &   # consider only the years between 2008 and 2016
                (df['date']!='2008-02-29') & (df['date']!='2012-02-29') & (df['date']!='2016-02-29')]   # remove the 29th of February
    
    # Create final dataframe, considering daily consumption and average weather conditions for every day
    groups = [df_cut['date'], df_cut['year'], df_cut['month'], df_cut['day_of_week'], df_cut['holiday']]
    ready_df = pd.DataFrame({'demand': df_cut['demand'].groupby(groups).sum(), 
                            'drybulb': df_cut['drybulb'].groupby(groups).mean(), 
                            'dewpnt': df_cut['dewpnt'].groupby(groups).mean()})
    ready_df = ready_df.reset_index(level = ['date','year', 'month', 'day_of_week', 'holiday'])
    
    return ready_df


def data_standardize(ready_df):
    
    # This function standardizes weather variables of ready_df, mapping them in [0,1], 
    # moreover it adds log_demand and standardize log_demand to the dataframe
    #
    # INPUT:
    # ready_df:     dataframe containing energy demand, weather variables and calendar variables
    #
    # OUTPUT:
    # df_std:       dataframe containing energy demand, energy log_demand, standardized energy log_demand, 
    #               standardized weather variables and calendar variables
    
    df_std = ready_df 
    
    # Standardize weather variables
    df_std.drybulb = standardize(df_std.drybulb)
    df_std.dewpnt = standardize(df_std.dewpnt)

    # Add variable log_demand
    log_demand = np.log(df_std.demand)
    df_std['log_demand'] = log_demand

    # Add variable standardized log_demand
    df_std['std_demand'] = standardize(log_demand)
    
    return df_std