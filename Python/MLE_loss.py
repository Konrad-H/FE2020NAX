# %%
import tensorflow.keras.backend as k
from tensorflow import gather
# from numpy import pi

# %% MLE loss function
# THe loss function is clipped because the logarithm gives issues when near to 0
# custom_loss is a standard version
# loss_strike gives the chance to alter the strike of the clip.

def loss_strike(strike=.008):
    # function that outputs a modified custom_loss
    # input: 
    #   strike: strike to clip the var
    # output: 
    #   MLE_loss: custom loss function
    #   y2var: function that translates y_pred to the var
    def y2var(y_pred):
        var = k.square(gather(y_pred,[1],axis=1))
        put_strike = None # .5**2
        var = k.clip(var, strike, put_strike)      
        return var  
    def MLE_loss(y_true, y_pred):
        true = gather(y_true,[0],axis=1)
        mean = gather(y_pred,[0],axis=1)
        var = y2var(y_pred)

        # k.log(2*pi)/2 is a constant
        log_L = k.log(var)/2+k.square(mean-true)/(2*var)
        
        return (1000)*k.mean(log_L,axis=-1)
    return MLE_loss, y2var
    