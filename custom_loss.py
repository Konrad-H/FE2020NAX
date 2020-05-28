import tensorflow.keras.backend as k
from numpy import pi

def custom_loss(y_true, y_pred):
    
    # calculate loss, using likehood function for residual Rt

    mean = y_pred[:,0]
    var = k.square(y_pred[:,1])

    log_L = -k.log(2*pi*var)/2-k.square(mean-y_true)/(2*var)
    return -(10**3)*k.mean(log_L)