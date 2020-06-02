def data_mining():
    
    import pandas as pd 
    import numpy as np
    from standardize_f import standardize 
    
    # %% Load Data
    df = pd.read_csv("C:/Users/User/Desktop/Università/Magistrale/Semestre 2/FE/Final project/gefcom.csv")
    # df.head()

    # %% Filter Data
    df_cut = df[(df['zone']=='TOTAL') & (df['year']>=2008) & (df['year']<=2016) & (df['date']!='2004-02-29') & (df['date']!='2008-02-29') & (df['date']!='2012-02-29') & (df['date']!='2016-02-29')]
    # df_cut.head()

    # %% Create final DF
    groups = [df_cut['date'], df_cut['year'], df_cut['month'], df_cut['day_of_week'], df_cut['holiday']]
    ready_df = pd.DataFrame({'demand': df_cut['demand'].groupby(groups).sum(),
                            'drybulb': df_cut['drybulb'].groupby(groups).mean(),
                            'dewpnt': df_cut['dewpnt'].groupby(groups).mean()})
    ready_df = ready_df.reset_index(level = ['date','year', 'month', 'day_of_week', 'holiday'])
    # ready_df.head()

    return ready_df


def data_standardize(ready_df):
    import numpy as np
    from standardize_f import standardize

    # %% Standardize DF
    df_std = ready_df 
    # [2] correction
    log_demand = np.log(df_std.demand)
    df_std['log_demand'] = log_demand
    df_std['std_demand'] = standardize(log_demand)
    df_std.drybulb = standardize(df_std.drybulb)
    df_std.dewpnt = standardize(df_std.dewpnt)
    df_std.head()

    return df_std