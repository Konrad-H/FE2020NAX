# %%
import tensorflow.keras.backend as k
from numpy import pi
import tensorflow as tf

# %%

def custom_loss(y_true, y_pred):
    
    strike = 1/(2*pi)/20 #Arbitary strike to make this work
    true= tf.gather(y_true,[0],axis=1)
    mean= tf.gather(y_pred,[0],axis=1)
    var = k.square(tf.gather(y_pred,[1],axis=1))
    var= k.clip(var, strike, None)


    log_L = -k.log(2*pi*var)/2-k.square(mean-true)/(2*var)
    return -(10000)*k.mean(log_L,axis=-1)

def custom_mse(y_true, y_pred):
    
    return -(10000)*k.mean(k.square(y_pred-y_true))

def inverse_std(demand_pred,demand_log_train):
    M = max(demand_log_train)
    m = min(demand_log_train)

    return demand_pred*(M-m)+m