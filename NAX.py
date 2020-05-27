# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
#import key packageskeras.utils.plot_model(model, "my_first_model.png")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import sys

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.linear_model import TweedieRegressor
from sklearn.linear_model import LinearRegression

import numpy as np
import pandas as pd 

import matplotlib as mpl
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

# %%
# DATA MINING

df = pd.read_csv("gefcom.csv") 
#print(data.head())
#print(data.tail())

## select data
df_tot = df[df['zone']=='TOTAL']
df_2 = df_tot[(df_tot['year']>=2009) & (df_tot['year']<=2016)] # ricordare di cambiare date 2011 -> 2016

#print(df_2.head())
#print(df_2.tail())


## create a DataFrame
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

df_years = pd.DataFrame({'date': date,
'demand': demand,
'drybulb': drybulb,
'dewpnt': dewpnt,
'year': year,
'month': month,
'day_of_week': day_of_week,
'holiday': holiday})
#data3 = pd.DataFrame({'date': date,'demand': demand,'drybulb': drybulb,'dewpnt': dewpnt,'year': year,'month': month,'day_of_week': day_of_week,'holiday': holiday})


# %%
print(df_years)


# %%

df_stand = df_years 
# [2] correction
N_df = len(df_stand.demand)
df_stand.demand = (df_stand.demand-[min(df_stand.demand)]*len(df_stand.demand))/(max(df_stand.demand)-min(df_stand.demand)) + [0.001]*len(df_stand.demand)
print(df_stand.demand)
print(np.mean(df_stand.demand))


# %%
# BUILD THE REGRESSORS

first_year = 2009
train_year = 2011

data = df_stand[(df_stand['year']>=2009) & (df_stand['year']<=train_year)]

log_consumption = np.log(data.demand)
omega = 2*np.pi/365
D_weekend = pd.get_dummies(data.day_of_week)
D_holiday = pd.get_dummies(data.holiday)
time_in_days = range(len(data))

# FIRST DAY IS march 1st, lacking 59 calendar days  6 years from 2003 to before 2009 6*365-59 number of days until 2009
time_since_dataset = (first_year-2004)*365-59

t=( np.array(time_in_days)+time_since_dataset )
# covariates
X=[t,np.sin(omega*t),np.cos(omega*t),np.sin(2*omega*t),np.cos(2*omega*t),D_weekend.Sat, D_weekend.Sun, D_holiday.iloc[:,1]] # manca hol
X = np.transpose(X)
#print(X)


# %%
# BASIC LINEAR REGRESSION

reg = LinearRegression()
reg.fit(X,log_consumption)
print('Linear')
print(reg.intercept_)
print(reg.coef_)
reg.score(X,log_consumption)


# %%
Xnew = [np.ones(len(X)),t,np.sin(omega*t),np.cos(omega*t),np.sin(2*omega*t),np.cos(2*omega*t),D_weekend.Sat, D_weekend.Sun, D_holiday.iloc[:,1]]
Xnew = np.transpose(Xnew)

gauss_log = sm.GLM(data.demand, Xnew, family=sm.families.Gaussian(sm.families.links.log))
gauss_log_results = gauss_log.fit()
param_new=gauss_log_results.params


# %%
# BETA DEL PROF
beta = [0.385, -0.000016, -0.003, -0.028, 0.136, -0.043, -0.146, -0.120, -0.060]


# %%
# PLOT GLM
plt.figure()


data.demand.plot()
inter = pd.Series(np.exp(np.array([reg.intercept_]*len(X))+np.dot(X,reg.coef_)))
inter.plot()

internew = pd.Series(np.exp(np.array(np.dot(Xnew,param_new))))
internew.plot()

plt.show()


# %%
#custom loss function
def custom_loss(y_true, y_pred):
    
    # calculate loss, using likehood function for residual Rt

    mean = y_pred[0]
    var = (y_pred[1])**2 
    loss = np.exp(-(y_true-mean)**2 /(2*var) ) / (2*np.pi*var)**.5
        
    return -10**3*tf.reduce_mean(np.log(loss), axis=-1)


# %%
#optimizer 
learn_rate = 0.1 #,.01, .003, .001
hidden_neurons = 6 # 3,4,5 
n_batch = 50 #none
act_f = "softmax" #'sigmoid'
reg_param = .001 #.0001, 0


# %%
data.head()


# %%
input_size = 10
output_states=2
look_back = 1


# %%
X = [data['drybulb'], data['dewpnt'], t, np.sin(omega*t), np.cos(omega*t), np.sin(2*omega*t), np.cos(2*omega*t),D_weekend.Sat, D_weekend.Sun, D_holiday.iloc[:,1]]


x_train= np.transpose(X)

len((x_train))


# %%
y_train = np.array(data['demand'])


# %%
model = keras.models.Sequential()

# NOT USED
act_reg = keras.regularizers.l1 (reg_param)


# %%
model.add(layers.LSTM ( 5 , input_shape=(input_size, look_back), return_sequences=True ))


# %%
# SIMPLE LSTM MODEL
model.add(layers.Dense(output_states))

# Optimizer
opt = keras.optimizers.Adam(learning_rate=learn_rate)
model.compile(loss='mae', optimizer=opt)


# %%

# FIT
model.fit(x_train, y_train, epochs=150, batch_size=n_batch , verbose=1)


# %%


