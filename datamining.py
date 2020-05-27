# DATA EXTRACION FROM gefcom

# %% packages
import numpy as np
import pandas as pd 
import os
import sys

# %% READ DF 
df = pd.read_csv("gefcom.csv") 
#print(data.head())
#print(data.tail())

# %% Filter Data 
df_agg_NE = df[df['zone']=='TOTAL'] #aggregated var by states
df_2 = df_agg_NE[(df_agg_NE['year']>=2008) & (df_agg_NE['year']<=2016)]


print(df_2.head())

# %% CREATE FINAL DF
date = df_2['date'].unique()
demand = np.zeros(len(date))
drybulb = np.zeros(len(date))
dewpnt = np.zeros(len(date))
year = np.zeros(len(date))
month = ['']*(len(date))
day_of_week = ['']*(len(date))
holiday = np.zeros(len(date),dtype=bool)
for n in range(len(date)):
    demand[n] = sum(df_2[df_2['date']==date[n]].demand)
    drybulb[n] = np.mean(df_2[df_2['date']==date[n]].drybulb)
    dewpnt[n] = np.mean(df_2[df_2['date']==date[n]].dewpnt)
    y = (df_2[df_2['date']==date[n]].year)
    year[n] = int(y.iloc[0])
    m = (df_2[df_2['date']==date[n]].month)
    month[n] = m.iloc[0]
    d = (df_2[df_2['date']==date[n]].day_of_week)
    day_of_week[n] = d.iloc[0]
    h = (df_2[df_2['date']==date[n]].holiday)
    holiday[n] = h.iloc[0]
    
ready_df = pd.DataFrame({'date': date,
                    'demand': demand,
                    'drybulb': drybulb,
                    'dewpnt': dewpnt,
                    'year': year,
                    'month': month,
                    'day_of_week': day_of_week,
                    'holiday': holiday})


# %% Standardize DF

df_std = ready_df 
# [2] correction
N_data = len(df_std.demand)
df_std.demand = (df_std.demand-[min(df_std.demand)]* 
                    N_data)/(max(df_std.demand)-
                    min(df_std.demand)) + [0.001]*N_data


# %% Write the DF

df_std.to_csv("GEFCON_standard.csv")