t2 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
c2=[20.,30.,40.,25.,46.,37.,65.7,23.,45.,67.,23.,34.5,35.2,23.]
day2 = ['M','T','W','H','F','Sat','Sun','M','T','W','H','F','Sat','Sun']
holiday2 = ['False','False','False','False','False','False','False','False','True','False','False','False','False','False']
    
data = {'ts': t2,
        'demand': c2,
        'day_of_week': day2,
        'holiday': holiday2}
import pandas as pd

data = pd.DataFrame (data, columns = ['ts','demand', 'day_of_week','holiday'])

def glm (data):

    import pandas as pd
    import numpy as np
    import os
    from sklearn import linear_model
    from sklearn.linear_model import LinearRegression
    import matplotlib.pyplot as plt

    #t = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
    #c=[20.,30.,40.,25.,46.,37.,65.7,23.,45.,67.,23.,34.5,35.2,23.]
    #day = ['M','T','W','H','F','Sat','Sun','M','T','W','H','F','Sat','Sun']
    #holiday = ['False','False','False','False','False','False','False','False','True','False','False','False','False','False']
    log_consumption = np.log(data.demand) #
    omega = 2*np.pi/365
    D_weekend = pd.get_dummies(data.day_of_week) #
    D_holiday = pd.get_dummies(data.holiday) #
    time_in_days = range(len(data))
    t=np.array(time_in_days)
    # covariates
    X=[t,np.sin(omega*t),np.cos(omega*t),np.sin(2*omega*t),np.cos(2*omega*t),D_weekend.Sat, D_weekend.Sun, D_holiday.iloc[:,1]] # manca hol
    X = np.transpose(X)

    #regressione con regressione lineare
    reg = LinearRegression()
    reg.fit(X,log_consumption)
    print(reg.intercept_)
    print(reg.coef_)

    # regressione con glm, peero viene di merda
   # from sklearn.linear_model import TweedieRegressor
   # glm = TweedieRegressor(power=0, alpha=0, link='identity')
   # glm.fit(X,log_consumption)
   # print(glm.intercept_)
   # print(glm.coef_)

    from sklearn.linear_model import TweedieRegressor
    glm2 = TweedieRegressor(power=0, alpha=0, link='log')
    glm2.fit(X,data.demand)
    print(glm2.intercept_)
    print(glm2.coef_)

    #plot
    plt.figure()
    
    data.demand.plot()
    inter = pd.Series(np.exp(np.array([reg.intercept_]*len(X))+np.dot(X,reg.coef_)))
    inter.plot()
    inter2 = pd.Series(np.exp(np.array([glm2.intercept_]*len(X))+np.dot(X,glm2.coef_)))
    inter2.plot()
    #inter1 = pd.Series(np.exp(np.array([glm.intercept_]*len(X))+np.dot(X,glm.coef_)))
    #inter1.plot()
    plt.show()

    coeff=[reg.intercept_,reg.coef_]
    return coeff
cf=glm(data)
print()
print(cf)
print()
print(cf[0])





