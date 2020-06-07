# %%
import tensorflow.keras.backend as k
from numpy import pi
import tensorflow as tf
import numpy as np

# %%

def custom_loss(y_true, y_pred):
    
    strike = 0.001
    
    true = tf.gather(y_true,[0],axis=1)
    mean = tf.gather(y_pred,[0],axis=1)
    var = k.square(tf.gather(y_pred,[1],axis=1))
    var = k.clip(var, strike, None)
    # var = k.square( strike**0.5+k.abs(tf.gather(y_pred,[1],axis=1)) )
    
    log_L = k.log(2*pi*var)/2+k.square(mean-true)/(2*var)
    
    return (1000)*k.mean(log_L,axis=-1)