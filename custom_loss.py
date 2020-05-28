# Custom 

from numpy import exp, pi

def custom_loss(y_true, y_pred):
    
    # calculate loss, using likehood function for residual Rt

    mean = y_pred[0]
    var = (y_pred[1])**2 
    loss = exp(-(y_true-mean)**2 /(2*var) ) / (2*pi*var)**.5
        
    return loss
    
  