# DATA EXTRACION FROM gefcom
 
import numpy as np
import pandas as pd 
import os
import sys
 
# data = pd.read_csv("example.csv") 
# data.head()
 
current_dir = os.path.dirname(__file__)
 
oneup_dir = os.path.join(current_dir, os.path.pardir)
 
file_path = os.path.join(oneup_dir, "./gefcom.csv")
data = pd.read_csv(file_path) 
#print(data.head())
#print(data.tail())
 
## select data
data1 = data[data['zone']=='TOTAL']
data2 = data1[(data1['year']>=2008) & (data1['year']<=2016)]
#print(data2.head())
#print(data2.tail())
print(data2.head())
 
## create a DataFrame
date = data2['date'].unique()
demand = np.zeros(len(date))
drybulb = np.zeros(len(date))
dewpnt = np.zeros(len(date))
year = np.zeros(len(date))
month = ['']*(len(date))
day_of_week = ['']*(len(date))
holiday = np.zeros(len(date),dtype=bool)
for n in range(len(date)):
    demand[n] = sum(data2[data2['date']==date[n]].demand)
    drybulb[n] = np.mean(data2[data2['date']==date[n]].drybulb)
    dewpnt[n] = np.mean(data2[data2['date']==date[n]].dewpnt)
    y = (data2[data2['date']==date[n]].year)
    year[n] = int(y.iloc[0])
    m = (data2[data2['date']==date[n]].month)
    month[n] = m.iloc[0]
    d = (data2[data2['date']==date[n]].day_of_week)
    day_of_week[n] = d.iloc[0]
    h = (data2[data2['date']==date[n]].holiday)
    holiday[n] = h.iloc[0]

data3 = pd.DataFrame({'date': date,
'demand': demand,
'drybulb': drybulb,
'dewpnt': dewpnt,
'year': year,
'month': month,
'day_of_week': day_of_week,
'holiday': holiday})
 
# 
# with  as f:

