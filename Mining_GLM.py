# DATA EXTRACION FROM gefcom

import numpy as np
import pandas as pd 
import os
import sys
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats

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
data2 = data1[(data1['year']>=2009) & (data1['year']<=2011)] # ricordare di cambiare date 2011 -> 2016
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
#data3 = pd.DataFrame({'date': date,'demand': demand,'drybulb': drybulb,'dewpnt': dewpnt,'year': year,'month': month,'day_of_week': day_of_week,'holiday': holiday})

print(data3)
data4 = data3
data4.demand = (data4.demand-[min(data4.demand)]*len(data4.demand))/(max(data4.demand)-min(data4.demand)) + [0.001]*len(data4.demand)
print(data4.demand)
print(np.mean(data4.demand))

#data4.drybulb = (data4.drybulb-[min(data4.drybulb)]*len(data4.drybulb))/(max(data4.drybulb)-min(data4.drybulb)) + [0.000001]*len(data4.drybulb)
#data4.dewpnt = (data4.dewpnt-[min(data4.dewpnt)]*len(data4.dewpnt))/(max(data4.dewpnt)-min(data4.dewpnt)) + [0.000001]*len(data4.dewpnt)
 
def glm (data):
    log_consumption = np.log(data.demand)
    omega = 2*np.pi/365
    D_weekend = pd.get_dummies(data.day_of_week)
    D_holiday = pd.get_dummies(data.holiday)
    time_in_days = range(len(data))
    t=np.array(time_in_days)
    # covariates
    X=[t,np.sin(omega*t),np.cos(omega*t),np.sin(2*omega*t),np.cos(2*omega*t),D_weekend.Sat, D_weekend.Sun, D_holiday.iloc[:,1]] # manca hol
    X = np.transpose(X)

    #regressione con regressione lineare
    reg = LinearRegression()
    reg.fit(X,log_consumption)
    print('Linear')
    print(reg.intercept_)
    print(reg.coef_)
    print()
    
    Xnew = [np.ones(len(X)),t,np.sin(omega*t),np.cos(omega*t),np.sin(2*omega*t),np.cos(2*omega*t),D_weekend.Sat, D_weekend.Sun, D_holiday.iloc[:,1]]
    Xnew = np.transpose(Xnew)
    print(len(Xnew))
    gauss_log = sm.GLM(data.demand, Xnew, family=sm.families.Gaussian(sm.families.links.log))
    gauss_log_results = gauss_log.fit()
    param_new=gauss_log_results.params
    print('GLM nuovo')
    print(gauss_log_results.params)
    print(gauss_log_results.summary())

    # regressione con glm, peero viene di merda
    from sklearn.linear_model import TweedieRegressor
    glm = TweedieRegressor(power=0, alpha=0, link='identity')
    glm.fit(X,log_consumption)
    print('GLM no link f.')
    print(glm.intercept_)
    print(glm.coef_)
    print()

    glm2 = TweedieRegressor(power=0, alpha=0, link='log')
    glm2.fit(X,data.demand)
    print('GLM fosse giusto')
    print(glm2.intercept_)
    print(glm2.coef_)
 
    beta = [0.385, -0.000016, -0.003, -0.028, 0.136, -0.043, -0.146, -0.120, -0.060]

#plot
    plt.figure()


    data.demand.plot()
    inter = pd.Series(np.exp(np.array([reg.intercept_]*len(X))+np.dot(X,reg.coef_)))
    inter.plot()
    #inter2 = pd.Series(np.exp(np.array([glm2.intercept_]*len(X))+np.dot(X,glm2.coef_)))
    #inter2.plot()
    inter1 = pd.Series(np.exp(np.array([glm.intercept_]*len(X))+np.dot(X,glm.coef_)))
    inter1.plot()
    internew = pd.Series(np.exp(np.array(np.dot(Xnew,param_new))))
    internew.plot()
    #interbav = pd.Series(np.exp(np.array(np.dot(Xnew,beta))))
    #interbav.plot()
    plt.show()
    coeff=[glm2.intercept_,glm2.coef_]
    return coeff

data5 = data4[(data4['year']>=2009) & (data4['year']<=2011)]
cf=glm(data5)

