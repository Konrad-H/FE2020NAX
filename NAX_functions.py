# %%
import tensorflow.keras.backend as k
from numpy import pi, sign
import tensorflow as tf

# %%
def y2var(y_pred):
    strike = 1/(2*pi)/20 #Arbitary strike to make this work
    strike = .0001
    var = k.square(tf.gather(y_pred,[1],axis=1))
    var= k.clip(var, strike, None)
    # var = k.square( strike**0.5+k.abs(tf.gather(y_pred,[1],axis=1)) )
    
    return var

def custom_loss(y_true, y_pred):


    true= tf.gather(y_true,[0],axis=1)
    mean= tf.gather(y_pred,[0],axis=1)
    var = y2var(y_pred)
    i=2
    if i==1:
        log_L = -k.log(2*pi)/2-k.log(var)/2-k.square(mean-true)/(2*var)
        return -(1000)*k.mean(log_L,axis=-1)
    elif i==2:
        log_L = -k.log(var)/2-.5/(2*var)
        return -(10000)*k.mean(log_L,axis=-1)

    elif i==3: 
        return 1000*k.square(var-1)

    

def custom_mse(y_true, y_pred):
    
    return -(10000)*k.mean(k.square(y_pred-y_true))

def inverse_std(demand_pred,demand_log_train):
    M = max(demand_log_train)
    m = min(demand_log_train)

    return demand_pred*(M-m)+m

# %%
