# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
#import key packageskeras.utils.plot_model(model, "my_first_model.png")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

from tf_ts_functions import  plot_train_history, multivariate_data
from NAX_functions import custom_loss, inverse_std
from tensorflow.keras.callbacks import EarlyStopping

from NAX_f import prep_data, aggregate_data, NAX_model, demands


mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False


# %% LOAD DATA
df_NAX = pd.read_csv("train_data.csv",index_col=0) 
df_NAX.head() #length 1095
## UNIVARIATE MODEL 
## TO UNDERSTAND THIS 

# %% PLOT AND STANDARDIZE
# RMSE_NAX 15776.347510314612 with sigmoid

START_SPLIT = 0
TRAIN_SPLIT = 1095
VAL_SPLIT = 1095+365
BATCH_SIZE = 50 #None
BUFFER_SIZE = 5

EVALUATION_INTERVAL = 500
EPOCHS = 300 #200
REG_PARAM = 0.0001
ACT_FUN = 'softmax' #'sigmoid' 'softmax'
LEARN_RATE = 0.003
HIDDEN_NEURONS=3 #3
LOSS_FUNCTION =  custom_loss #custom_loss #'mae', 'mse'
OUTPUT_NEURONS= 2 #2
STOPPATIENCE = 10

past_history = 2
future_target = -1
STEP = 1

# opt=tf.keras.optimizers.RMSprop()
tf.random.set_seed(14)
features,labels= prep_data(df_NAX,
                    START_SPLIT = START_SPLIT,
                    TRAIN_SPLIT = TRAIN_SPLIT,
                    VAL_SPLIT = VAL_SPLIT)
x_train, y_train,x_val, y_val = aggregate_data(features,labels,
                                  START_SPLIT = START_SPLIT,
                                  TRAIN_SPLIT = TRAIN_SPLIT,
                                  VAL_SPLIT = VAL_SPLIT,
                                  past_history = past_history,
                                  future_target = future_target,
                                  STEP = STEP)


print ('Single window of past history : {}'.format(x_train[0].shape))

# %%

model = NAX_model(INPUT_SHAPE=x_train.shape[-2:],
            REG_PARAM = REG_PARAM,
            ACT_FUN = ACT_FUN,
            LEARN_RATE = LEARN_RATE,
            HIDDEN_NEURONS=HIDDEN_NEURONS ,
            OUTPUT_NEURONS= OUTPUT_NEURONS,
            LOSS_FUNCTION =  LOSS_FUNCTION)


# %%

# %%
EARLYSTOP = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=STOPPATIENCE)
history=model.fit(
    x=x_train, y=y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, callbacks=[EARLYSTOP],
    validation_data=(x_val,y_val), validation_batch_size=BATCH_SIZE,shuffle=True
)

plot_train_history(history,"Loss of model")
# %%
y_pred =model.predict(x_val)
START = TRAIN_SPLIT+past_history+future_target
demand_true, demand_NAX, demand_GLM  = demands(y_pred,y_val, df_NAX,START)

plt.figure()
demand_true.plot()
demand_NAX.plot()
demand_GLM.plot()
plt.show()

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
print('RMSE_GLM',rmse(demand_GLM, demand_true))
print('RMSE_NAX',rmse(demand_NAX, demand_true))