import tensorflow.keras.backend as k
from numpy import pi

def custom_loss(y_true, y_pred):
    
    # calculate loss, using likehood function for residual Rt
    #strike = 1/(2*pi)
    strike = 0.00001
    mean=k.clip(y_pred[:,0], -1,1)
    var=k.clip((y_pred[:,1]), strike, None)

    log_L = -k.log(2*pi*var)/2-k.square(mean-y_true)/(2*var)
    return -(10000)*k.mean(log_L)